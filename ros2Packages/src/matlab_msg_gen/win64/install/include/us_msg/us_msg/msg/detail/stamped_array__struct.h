// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice

#ifndef US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_H_
#define US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'array'
#include "std_msgs/msg/detail/float32_multi_array__struct.h"

/// Struct defined in msg/StampedArray in the package us_msg.
typedef struct us_msg__msg__StampedArray
{
  builtin_interfaces__msg__Time stamp;
  std_msgs__msg__Float32MultiArray array;
} us_msg__msg__StampedArray;

// Struct for a sequence of us_msg__msg__StampedArray.
typedef struct us_msg__msg__StampedArray__Sequence
{
  us_msg__msg__StampedArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} us_msg__msg__StampedArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_H_
