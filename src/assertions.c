#include <stdio.h>

void assert_known_bits(unsigned long long value, unsigned long long known, unsigned long long mask) {
    if ((value ^ known) & mask) {
        printf("ERROR: value=%llx, known=%llx, mask=%llx\n", value, known, mask);
    }
}

// int main(){
//     test(0x000000EE, 0x000000EE, 0x0000FFFF); // matches 100%
//     test(0x000000FF, 0x000000F0, 0x00000FF0); // matches mask
//     test(0x000000FF, 0x000000EF, 0x0000FFFF); // mismatch bit 7
//     return 0;
// }