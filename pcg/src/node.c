#include "dmpf64/node.h"
int dmpf64_node_cmp(dmpf64_node a, dmpf64_node b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}
