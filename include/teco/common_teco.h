#ifndef __COMMON_TECO_H__
#define __COMMON_TECO_H__

#include <sdaa_runtime.h>
#include <tecodnn.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_TECODNN(expression)                                                               \
    {                                                                                           \
        tecodnnStatus_t status = (expression);                                                  \
        if (status != TECODNN_STATUS_SUCCESS)                                                   \
        {                                                                                       \
            fprintf(stderr, "Error at line %d: %s\n", __LINE__, tecodnnGetErrorString(status)); \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    }

#define CHECK_TECOBLAS(expression)                                                               \
    {                                                                                            \
        tecoblasStatus_t status = (expression);                                                  \
        if (status != TECOBLAS_STATUS_SUCCESS)                                                   \
        {                                                                                        \
            fprintf(stderr, "Error at line %d: %s\n", __LINE__, tecoblasGetErrorString(status)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

#endif // __COMMON_TECO_H__
