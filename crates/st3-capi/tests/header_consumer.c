/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026, The SourceTracker3 Development Team */

/*
 * A minimal consumer of st3.h, compiled (never linked or run) by
 * tests/header.rs.
 *
 * Arrow's abi.h and nanoarrow declare the C Data Interface structs by tag only
 * (`struct ArrowSchema`, no typedef), and a consumer may include them before
 * st3.h, after it, or not at all. header.rs compiles this file with -Werror in
 * each of those orders (-DARROW_ABI_BEFORE, -DARROW_ABI_AFTER, neither), as C
 * and as C++.
 */

#ifdef ARROW_ABI_BEFORE
#include "arrow_abi.h"
#endif

#include "st3.h"

#ifdef ARROW_ABI_AFTER
#include "arrow_abi.h"
#endif

int main(void) {
  /* Consumer-side pointers to every Arrow struct st3.h takes. If st3.h named
   * a type other than the file-scope `struct Arrow...` these declare, the calls
   * below would pass incompatible pointers. */
  struct ArrowArray *array = NULL;
  struct ArrowSchema *schema = NULL;
  struct ArrowArrayStream *stream = NULL;
  struct St3Table *table = NULL;
  const struct St3Result *result = NULL;

  (void)st3_table_from_arrow(array, schema, array, schema, array, schema,
                             &table);
  (void)st3_result_means(result, array, schema);
  (void)st3_result_stds(result, array, schema);
  (void)st3_result_contingency_stream(result, stream);
  return 0;
}
