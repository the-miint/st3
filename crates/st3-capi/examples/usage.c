/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026, The SourceTracker3 Development Team */

/*
 * A worked example of the SourceTracker3 C ABI, end to end.
 *
 * It walks a downstream C consumer through the whole flow:
 *
 *   1. query the ABI version (st3_abi_version);
 *   2. hand SourceTracker3 a dataset built as Arrow C Data Interface structures
 *      -- a COO count table, a feature-id array, and a per-sample metadata table
 *      -- via st3_table_from_arrow;
 *   3. run source attribution with a versioned St3Config (st3_run);
 *   4. read back the dense mixing means (st3_result_means) and standard
 *      deviations (st3_result_stds);
 *   5. drain the per-sink source x taxon assignment tally, which arrives as an
 *      Arrow array stream (st3_result_contingency_stream);
 *   6. check every return code against St3Status and print st3_last_error on
 *      failure; and
 *   7. free every handle and release every exported Arrow structure.
 *
 * Build it with `make header` (or `cargo build --release`) to emit st3.h and
 * the shared library, then, from the repository root:
 *
 *   cc -std=c11 -I target crates/st3-capi/examples/usage.c \
 *       -L target/release -lst3 -lm -o usage
 *   LD_LIBRARY_PATH=target/release ./usage
 *
 * (`-I target` is where `make header` copies st3.h; point it at wherever your
 * header lives. `crates/st3-capi/tests/c_example.rs` compiles and runs this file
 * as part of the test suite.)
 *
 * The input buffers below are static, so the input arrays carry no-op release
 * callbacks -- they own nothing to free. The arrays and streams that the library
 * exports are owned by SourceTracker3's own Arrow allocations and are released
 * through their own release callbacks.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * The Arrow C Data Interface structs (the standardized ABI). They are declared
 * here -- as typedefs matching the names st3.h refers to -- so the header can be
 * included immediately below. A real consumer typically gets these from Arrow's
 * own `abi.h` instead.
 */
typedef struct ArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;
  void (*release)(struct ArrowSchema *);
  void *private_data;
} ArrowSchema;

typedef struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  struct ArrowArray *dictionary;
  void (*release)(struct ArrowArray *);
  void *private_data;
} ArrowArray;

typedef struct ArrowArrayStream {
  int (*get_schema)(struct ArrowArrayStream *, struct ArrowSchema *);
  int (*get_next)(struct ArrowArrayStream *, struct ArrowArray *);
  const char *(*get_last_error)(struct ArrowArrayStream *);
  void (*release)(struct ArrowArrayStream *);
  void *private_data;
} ArrowArrayStream;

#include "st3.h"

/* Arrow schema flag: the field is nullable. */
#define ARROW_FLAG_NULLABLE 2

/* No-op release callbacks for our static input buffers. Per the C Data
 * Interface, a release callback marks its object released by nulling its own
 * release pointer. */
static void noop_release_array(ArrowArray *a) { a->release = NULL; }
static void noop_release_schema(ArrowSchema *s) { s->release = NULL; }

/* ---- The dataset --------------------------------------------------------------
 * 3 features (f0, f1, f2) and 4 samples: two sources (src_a in envA, src_b in
 * envB) and two sinks (sink0, sink1). Ten COO entries, each (row = feature index,
 * col = sample index, val = count). sink0 leans on f0 (envA's marker) and sink1
 * on f2 (envB's marker), so we expect sink0 to attribute mostly to envA and
 * sink1 mostly to envB.
 */
enum { NNZ = 10, N_SAMPLES = 4, N_FEATURES = 3, N_SINKS = 2, N_ENVS = 3 };

/* COO children: parallel row / col / val arrays. */
static int32_t coo_row[NNZ] = {0, 1, 1, 2, 0, 1, 2, 0, 1, 2};
static int32_t coo_col[NNZ] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3};
static double coo_val[NNZ] = {10, 1, 1, 10, 8, 1, 1, 1, 1, 8};

/* An Arrow primitive array has two buffers: a validity bitmap (NULL here, since
 * nothing is null) and the values. */
static const void *coo_row_bufs[2] = {NULL, coo_row};
static const void *coo_col_bufs[2] = {NULL, coo_col};
static const void *coo_val_bufs[2] = {NULL, coo_val};

static ArrowArray coo_row_arr = {NNZ, 0, 0, 2, 0, coo_row_bufs, NULL, NULL,
                                 noop_release_array, NULL};
static ArrowArray coo_col_arr = {NNZ, 0, 0, 2, 0, coo_col_bufs, NULL, NULL,
                                 noop_release_array, NULL};
static ArrowArray coo_val_arr = {NNZ, 0, 0, 2, 0, coo_val_bufs, NULL, NULL,
                                 noop_release_array, NULL};
static ArrowArray *coo_children[3] = {&coo_row_arr, &coo_col_arr, &coo_val_arr};
/* A struct array has a single (validity) buffer and one child per field. */
static const void *coo_struct_bufs[1] = {NULL};
static ArrowArray coo_struct_arr = {NNZ, 0, 0, 1, 3, coo_struct_bufs,
                                    coo_children, NULL, noop_release_array, NULL};

/* Schemas mirror the arrays. Format "i" = int32, "g" = float64, "+s" = struct. */
static ArrowSchema coo_row_schema = {"i",  "row", NULL, 0,
                                     0,    NULL,  NULL, noop_release_schema,
                                     NULL};
static ArrowSchema coo_col_schema = {"i",  "col", NULL, 0,
                                     0,    NULL,  NULL, noop_release_schema,
                                     NULL};
static ArrowSchema coo_val_schema = {"g",  "val", NULL, 0,
                                     0,    NULL,  NULL, noop_release_schema,
                                     NULL};
static ArrowSchema *coo_schema_children[3] = {&coo_row_schema, &coo_col_schema,
                                              &coo_val_schema};
static ArrowSchema coo_schema = {"+s", NULL, NULL, 0,
                                 3,    coo_schema_children, NULL,
                                 noop_release_schema, NULL};

/* feature_ids: ["f0", "f1", "f2"] as Utf8 (format "u"): three buffers -- validity
 * (NULL), int32 offsets, and packed character data. */
static int32_t fid_offsets[N_FEATURES + 1] = {0, 2, 4, 6};
static char fid_data[6] = "f0f1f2";
static const void *fid_bufs[3] = {NULL, fid_offsets, fid_data};
static ArrowArray fid_arr = {N_FEATURES, 0, 0, 3, 0, fid_bufs, NULL, NULL,
                             noop_release_array, NULL};
static ArrowSchema fid_schema = {"u",  NULL, NULL, 0,
                                 0,    NULL, NULL, noop_release_schema, NULL};

/* Per-sample metadata: sample_id / role / env (env nullable). */
static int32_t sid_offsets[N_SAMPLES + 1] = {0, 5, 10, 15, 20};
static char sid_data[20] = "src_asrc_bsink0sink1";
static const void *sid_bufs[3] = {NULL, sid_offsets, sid_data};
static ArrowArray sid_arr = {N_SAMPLES, 0, 0, 3, 0, sid_bufs, NULL, NULL,
                             noop_release_array, NULL};

static int32_t role_offsets[N_SAMPLES + 1] = {0, 6, 12, 16, 20};
static char role_data[20] = "sourcesourcesinksink";
static const void *role_bufs[3] = {NULL, role_offsets, role_data};
static ArrowArray role_arr = {N_SAMPLES, 0, 0, 3, 0, role_bufs, NULL, NULL,
                              noop_release_array, NULL};

/* env = ["envA", "envB", NULL, NULL]: the sinks have no environment. The two
 * present values set validity bits 0b0011; null_count is 2. */
static uint8_t env_validity[1] = {0x03};
static int32_t env_offsets[N_SAMPLES + 1] = {0, 4, 8, 8, 8};
static char env_data[8] = "envAenvB";
static const void *env_bufs[3] = {env_validity, env_offsets, env_data};
static ArrowArray env_arr = {N_SAMPLES, 2, 0, 3, 0, env_bufs, NULL, NULL,
                             noop_release_array, NULL};

static ArrowArray *meta_children[3] = {&sid_arr, &role_arr, &env_arr};
static const void *meta_struct_bufs[1] = {NULL};
static ArrowArray meta_struct_arr = {N_SAMPLES, 0, 0, 1, 3, meta_struct_bufs,
                                     meta_children, NULL, noop_release_array,
                                     NULL};

static ArrowSchema sid_schema = {"u",  "sample_id", NULL, 0,
                                 0,    NULL,        NULL, noop_release_schema,
                                 NULL};
static ArrowSchema role_schema = {"u",  "role", NULL, 0,
                                  0,    NULL,   NULL, noop_release_schema,
                                  NULL};
static ArrowSchema env_schema = {"u",  "env", NULL, ARROW_FLAG_NULLABLE,
                                 0,    NULL,  NULL, noop_release_schema,
                                 NULL};
static ArrowSchema *meta_schema_children[3] = {&sid_schema, &role_schema,
                                               &env_schema};
static ArrowSchema meta_schema = {"+s", NULL, NULL, 0,
                                  3,    meta_schema_children, NULL,
                                  noop_release_schema, NULL};

/* Release an exported (ArrowArray, ArrowSchema) pair through their own
 * callbacks. Safe to call whether or not the library populated them. */
static void release_pair(ArrowArray *arr, ArrowSchema *schema) {
  if (arr->release != NULL) {
    arr->release(arr);
  }
  if (schema->release != NULL) {
    schema->release(schema);
  }
}

/* Print each sink's dominant source environment from a dense means/stds batch.
 *
 * The batch is a struct array: `length` rows (one per sink) and `n_children`
 * columns -- column 0 is the Utf8 `sink_id`, and columns 1.. are one Float64 per
 * environment. The environment names live in the schema's child field names.
 * For a Float64 column, buffers[1] is the values (honour the array's own offset,
 * which is 0 for a freshly exported batch but defensive if it is ever sliced).
 */
static void summarize_means(const ArrowArray *arr, const ArrowSchema *schema) {
  int64_t n_sinks = arr->length;
  int64_t n_cols = arr->n_children;

  const ArrowArray *id_col = arr->children[0];
  const int32_t *id_off = (const int32_t *)id_col->buffers[1] + id_col->offset;
  const char *id_data = (const char *)id_col->buffers[2];

  for (int64_t r = 0; r < n_sinks; r++) {
    int64_t best = 1;
    double best_val = -1.0;
    for (int64_t c = 1; c < n_cols; c++) {
      const ArrowArray *col = arr->children[c];
      double v = ((const double *)col->buffers[1])[col->offset + r];
      if (v > best_val) {
        best_val = v;
        best = c;
      }
    }
    int32_t start = id_off[r];
    int32_t end = id_off[r + 1];
    const char *env_name = schema->children[best]->name;
    printf("  %.*s: mostly %s (%.3f)\n", (int)(end - start), id_data + start,
           env_name, best_val);
  }
}

int main(void) {
  /* 1. ABI version. This is a compile-time constant the consumer can gate on. */
  printf("SourceTracker3 C ABI version %u\n", st3_abi_version());

  /* 2. Import the dataset over the Arrow C Data Interface. On success the three
   * input arrays are consumed by the library; we must not release them. */
  St3Table *table = NULL;
  St3Status st = st3_table_from_arrow(&coo_struct_arr, &coo_schema, &fid_arr,
                                      &fid_schema, &meta_struct_arr,
                                      &meta_schema, &table);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "import failed (%d): %s\n", (int)st, st3_last_error());
    return 1;
  }

  /* 3. Configure a deterministic sink run. Zero the struct first so any field we
   * do not set (e.g. rarefaction depths) is a well-defined 0, then stamp the
   * version/size prefix and the parameters. Contingency is on so we can drain the
   * assignment stream in step 5. */
  St3Config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_version = ST3_CONFIG_V1;
  cfg.struct_size = (uint32_t)sizeof(St3Config);
  cfg.seed = 42;
  cfg.jobs = 1; /* 0 = all cores, 1 = serial, n = n threads */
  cfg.collapse = ST3_COLLAPSE_SUM;
  cfg.estimator = ST3_ESTIMATOR_KIND_GIBBS_COLLAPSED;
  cfg.contingency = 1;
  cfg.alpha1 = 0.001;
  cfg.alpha2 = 0.1;
  cfg.beta = 10.0;
  cfg.restarts = 20;
  cfg.draws_per_restart = 5;
  cfg.burnin = 10;
  cfg.delay = 1;

  St3Result *result = NULL;
  st = st3_run(table, &cfg, &result);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "run failed (%d): %s\n", (int)st, st3_last_error());
    st3_table_free(table);
    return 1;
  }

  /* 4a. Export the dense mixing means and summarize them. */
  ArrowArray means_arr;
  ArrowSchema means_schema;
  st = st3_result_means(result, &means_arr, &means_schema);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "means export failed (%d): %s\n", (int)st, st3_last_error());
    st3_result_free(result);
    st3_table_free(table);
    return 1;
  }
  printf("mixing means (%lld sinks x %lld env columns):\n",
         (long long)means_arr.length, (long long)(means_arr.n_children - 1));
  summarize_means(&means_arr, &means_schema);
  release_pair(&means_arr, &means_schema);

  /* 4b. Export the dense per-draw standard deviations (same shape as the means).
   * We just confirm the shape here; a real consumer would read them like the
   * means above. */
  ArrowArray stds_arr;
  ArrowSchema stds_schema;
  st = st3_result_stds(result, &stds_arr, &stds_schema);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "stds export failed (%d): %s\n", (int)st, st3_last_error());
    st3_result_free(result);
    st3_table_free(table);
    return 1;
  }
  printf("mixing stds:  %lld sinks x %lld env columns\n",
         (long long)stds_arr.length, (long long)(stds_arr.n_children - 1));
  release_pair(&stds_arr, &stds_schema);

  /* 5. Drain the per-sink contingency stream: one flat-COO batch per sink over
   * the schema [sink, source, feature, value]. We count the batches and sum the
   * `value` column (child index 3); the total mass equals the sinks' summed
   * depth. Draining ends when get_next hands back a released (null-release)
   * array. */
  ArrowArrayStream stream;
  st = st3_result_contingency_stream(result, &stream);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "contingency stream failed (%d): %s\n", (int)st,
            st3_last_error());
    st3_result_free(result);
    st3_table_free(table);
    return 1;
  }

  int n_batches = 0;
  double contingency_mass = 0.0;
  for (;;) {
    ArrowArray batch;
    if (stream.get_next(&stream, &batch) != 0) {
      const char *msg = stream.get_last_error ? stream.get_last_error(&stream)
                                              : "unknown stream error";
      fprintf(stderr, "contingency get_next failed: %s\n", msg);
      stream.release(&stream);
      st3_result_free(result);
      st3_table_free(table);
      return 1;
    }
    if (batch.release == NULL) {
      break; /* end of stream */
    }
    const ArrowArray *value_col = batch.children[3];
    const double *values =
        (const double *)value_col->buffers[1] + value_col->offset;
    for (int64_t i = 0; i < batch.length; i++) {
      contingency_mass += values[i];
    }
    n_batches++;
    batch.release(&batch);
  }
  stream.release(&stream);
  printf("contingency: %d per-sink batches, total assigned mass %.1f\n",
         n_batches, contingency_mass);

  /* 6/7. last_error is null after a successful run; free every handle. */
  if (st3_last_error() != NULL) {
    fprintf(stderr, "unexpected last_error after success: %s\n",
            st3_last_error());
    st3_result_free(result);
    st3_table_free(table);
    return 1;
  }
  st3_result_free(result);
  st3_table_free(table);

  printf("usage example OK\n");
  return 0;
}
