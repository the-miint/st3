/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026, The SourceTracker3 Development Team */

/*
 * End-to-end C harness for the SourceTracker3 C ABI.
 *
 * It hand-builds a small dataset as Arrow C Data Interface structures (a COO
 * struct array, a feature-id string array, and a metadata struct array), then
 * drives st3_table_from_arrow -> st3_run -> st3_result_means, reads the exported
 * dense means back, and checks the shape and that each sink row sums to 1. It
 * also checks that st3_last_error() is null on success and non-null after a
 * forced error, and frees every handle.
 *
 * All input buffers are static, so the input arrays use no-op release callbacks
 * (they own nothing to free). The exported means array is owned by the library's
 * arrow allocation and is released via its own release callback.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Arrow C Data Interface (the standardized ABI structs). Defined here — as
 * typedefs, matching the names st3.h uses — so the header can be included below.
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

/* Arrow schema flag: field is nullable. */
#define ARROW_FLAG_NULLABLE 2

/* No-op release callbacks: the input buffers are static, nothing to free. Per
 * the C Data Interface, a release callback marks the object released by nulling
 * its own release pointer. */
static void noop_release_array(ArrowArray *a) { a->release = NULL; }
static void noop_release_schema(ArrowSchema *s) { s->release = NULL; }

/* ---- Dataset -----------------------------------------------------------------
 * 3 features (f0,f1,f2); 4 samples: src_a (source, envA), src_b (source, envB),
 * sink0 (sink), sink1 (sink). 10 COO entries (row=feature, col=sample, val=count).
 */
enum { NNZ = 10, N_SAMPLES = 4, N_FEATURES = 3, N_SINKS = 2, N_ENVS = 3 };

/* COO children. */
static int32_t coo_row[NNZ] = {0, 1, 1, 2, 0, 1, 2, 0, 1, 2};
static int32_t coo_col[NNZ] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3};
static double coo_val[NNZ] = {10, 1, 1, 10, 8, 1, 1, 1, 1, 8};

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
static const void *coo_struct_bufs[1] = {NULL};
static ArrowArray coo_struct_arr = {NNZ, 0, 0, 1, 3, coo_struct_bufs,
                                    coo_children, NULL, noop_release_array, NULL};

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

/* feature_ids: ["f0","f1","f2"] (Utf8). */
static int32_t fid_offsets[N_FEATURES + 1] = {0, 2, 4, 6};
static char fid_data[6] = "f0f1f2";
static const void *fid_bufs[3] = {NULL, fid_offsets, fid_data};
static ArrowArray fid_arr = {N_FEATURES, 0, 0, 3, 0, fid_bufs, NULL, NULL,
                             noop_release_array, NULL};
static ArrowSchema fid_schema = {"u",  NULL, NULL, 0,
                                 0,    NULL, NULL, noop_release_schema, NULL};

/* metadata: sample_id / role / env (env nullable). */
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

/* env = ["envA","envB",NULL,NULL]: validity bits 0b0011, two nulls. */
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

int main(void) {
  /* Import. */
  St3Table *table = NULL;
  St3Status st = st3_table_from_arrow(&coo_struct_arr, &coo_schema, &fid_arr,
                                      &fid_schema, &meta_struct_arr,
                                      &meta_schema, &table);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "import failed (%d): %s\n", (int)st, st3_last_error());
    return 1;
  }
  if (st3_last_error() != NULL) {
    fprintf(stderr, "last_error should be null after a successful import\n");
    return 1;
  }
  if (table == NULL) {
    fprintf(stderr, "import returned OK but a null handle\n");
    return 1;
  }

  /* Configure a deterministic sink run (sum collapse, matching the fixtures). */
  St3Config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_version = ST3_CONFIG_V1;
  cfg.struct_size = (uint32_t)sizeof(St3Config);
  cfg.seed = 42;
  cfg.jobs = 1;
  cfg.collapse = ST3_COLLAPSE_SUM;
  cfg.estimator = ST3_ESTIMATOR_KIND_GIBBS_COLLAPSED;
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
    return 1;
  }

  /* Export the dense means and inspect them. */
  ArrowArray means_arr;
  ArrowSchema means_schema;
  st = st3_result_means(result, &means_arr, &means_schema);
  if (st != ST3_STATUS_OK) {
    fprintf(stderr, "means export failed (%d): %s\n", (int)st, st3_last_error());
    return 1;
  }

  /* The batch is a struct array: length = rows (sinks); n_children = columns
   * (sink_id + one Float64 per environment). */
  int64_t n_sinks = means_arr.length;
  int64_t n_cols = means_arr.n_children;
  if (n_sinks != N_SINKS) {
    fprintf(stderr, "expected %d sinks, got %lld\n", N_SINKS, (long long)n_sinks);
    return 1;
  }
  if (n_cols != 1 + N_ENVS) {
    fprintf(stderr, "expected %d columns, got %lld\n", 1 + N_ENVS,
            (long long)n_cols);
    return 1;
  }

  /* Checksum: each sink row must sum to 1 across the env columns (1..n_cols);
   * the grand total is therefore n_sinks. */
  double total = 0.0;
  for (int64_t r = 0; r < n_sinks; r++) {
    double row_sum = 0.0;
    for (int64_t c = 1; c < n_cols; c++) {
      const ArrowArray *col = means_arr.children[c];
      /* Float64 layout: buffers[1] is the data buffer; honor the array's own
       * offset (0 for a freshly exported batch, but defensive if it is ever
       * sliced). */
      const double *vals = (const double *)col->buffers[1] + col->offset;
      row_sum += vals[r];
    }
    if (fabs(row_sum - 1.0) > 1e-9) {
      fprintf(stderr, "sink %lld row sums to %f, expected 1.0\n", (long long)r,
              row_sum);
      return 1;
    }
    total += row_sum;
  }
  if (fabs(total - (double)N_SINKS) > 1e-9) {
    fprintf(stderr, "checksum %f, expected %f\n", total, (double)N_SINKS);
    return 1;
  }

  /* Release the library-owned exported means. */
  if (means_arr.release != NULL) {
    means_arr.release(&means_arr);
  }
  if (means_schema.release != NULL) {
    means_schema.release(&means_schema);
  }

  /* Forced error: a null result handle must fail and set the last error. */
  ArrowArray err_arr;
  ArrowSchema err_schema;
  St3Status err = st3_result_means(NULL, &err_arr, &err_schema);
  if (err == ST3_STATUS_OK) {
    fprintf(stderr, "expected an error from a null result handle\n");
    return 1;
  }
  if (st3_last_error() == NULL) {
    fprintf(stderr, "expected last_error to be set after a forced error\n");
    return 1;
  }

  st3_result_free(result);
  st3_table_free(table);

  printf("harness OK: %lld sinks x %lld cols, checksum=%f\n", (long long)n_sinks,
         (long long)n_cols, total);
  return 0;
}
