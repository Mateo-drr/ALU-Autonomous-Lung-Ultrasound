// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "us_msg/msg/detail/stamped_array__rosidl_typesupport_introspection_c.h"
#include "us_msg/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "us_msg/msg/detail/stamped_array__functions.h"
#include "us_msg/msg/detail/stamped_array__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `array`
#include "std_msgs/msg/float32_multi_array.h"
// Member `array`
#include "std_msgs/msg/detail/float32_multi_array__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  us_msg__msg__StampedArray__init(message_memory);
}

void us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_fini_function(void * message_memory)
{
  us_msg__msg__StampedArray__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_member_array[2] = {
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(us_msg__msg__StampedArray, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "array",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(us_msg__msg__StampedArray, array),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_members = {
  "us_msg__msg",  // message namespace
  "StampedArray",  // message name
  2,  // number of fields
  sizeof(us_msg__msg__StampedArray),
  us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_member_array,  // message members
  us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_init_function,  // function to initialize message memory (memory has to be allocated)
  us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_type_support_handle = {
  0,
  &us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_us_msg
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, us_msg, msg, StampedArray)() {
  us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float32MultiArray)();
  if (!us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_type_support_handle.typesupport_identifier) {
    us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &us_msg__msg__StampedArray__rosidl_typesupport_introspection_c__StampedArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
