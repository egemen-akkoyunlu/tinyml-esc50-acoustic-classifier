#include "em_ldma.h"
void test() {
    LDMA_Descriptor_t d = LDMA_DESCRIPTOR_LINKREL_P2M_HALF(&(USART0->RXDATA), 0, 10, 1);
}
