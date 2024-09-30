// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice

#ifndef US_MSG__MSG__DETAIL__STAMPED_ARRAY__FUNCTIONS_H_
#define US_MSG__MSG__DETAIL__STAMPED_ARRAY__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "us_msg/msg/rosidl_generator_c__visibility_control.h"

#include "us_msg/msg/detail/stamped_array__struct.h"

/// Initialize msg/StampedArray message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * us_msg__msg__StampedArray
 * )) before or use
 * us_msg__msg__StampedArray__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__init(us_msg__msg__StampedArray * msg);

/// Finalize msg/StampedArray message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
void
us_msg__msg__StampedArray__fini(us_msg__msg__StampedArray * msg);

/// Create msg/StampedArray message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * us_msg__msg__StampedArray__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
us_msg__msg__StampedArray *
us_msg__msg__StampedArray__create();

/// Destroy msg/StampedArray message.
/**
 * It calls
 * us_msg__msg__StampedArray__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
void
us_msg__msg__StampedArray__destroy(us_msg__msg__StampedArray * msg);

/// Check for msg/StampedArray message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__are_equal(const us_msg__msg__StampedArray * lhs, const us_msg__msg__StampedArray * rhs);

/// Copy a msg/StampedArray message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__copy(
  const us_msg__msg__StampedArray * input,
  us_msg__msg__StampedArray * output);

/// Initialize array of msg/StampedArray messages.
/**
 * It allocates the memory for the number of elements and calls
 * us_msg__msg__StampedArray__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__Sequence__init(us_msg__msg__StampedArray__Sequence * array, size_t size);

/// Finalize array of msg/StampedArray messages.
/**
 * It calls
 * us_msg__msg__StampedArray__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
void
us_msg__msg__StampedArray__Sequence__fini(us_msg__msg__StampedArray__Sequence * array);

/// Create array of msg/StampedArray messages.
/**
 * It allocates the memory for the array and calls
 * us_msg__msg__StampedArray__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
us_msg__msg__StampedArray__Sequence *
us_msg__msg__StampedArray__Sequence__create(size_t size);

/// Destroy array of msg/StampedArray messages.
/**
 * It calls
 * us_msg__msg__StampedArray__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
void
us_msg__msg__StampedArray__Sequence__destroy(us_msg__msg__StampedArray__Sequence * array);

/// Check for msg/StampedArray message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__Sequence__are_equal(const us_msg__msg__StampedArray__Sequence * lhs, const us_msg__msg__StampedArray__Sequence * rhs);

/// Copy an array of msg/StampedArray messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_us_msg
bool
us_msg__msg__StampedArray__Sequence__copy(
  const us_msg__msg__StampedArray__Sequence * input,
  us_msg__msg__StampedArray__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // US_MSG__MSG__DETAIL__STAMPED_ARRAY__FUNCTIONS_H_
