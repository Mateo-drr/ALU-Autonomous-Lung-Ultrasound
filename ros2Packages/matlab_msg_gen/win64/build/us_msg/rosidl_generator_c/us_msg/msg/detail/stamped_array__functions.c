// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice
#include "us_msg/msg/detail/stamped_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `array`
#include "std_msgs/msg/detail/float32_multi_array__functions.h"

bool
us_msg__msg__StampedArray__init(us_msg__msg__StampedArray * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    us_msg__msg__StampedArray__fini(msg);
    return false;
  }
  // array
  if (!std_msgs__msg__Float32MultiArray__init(&msg->array)) {
    us_msg__msg__StampedArray__fini(msg);
    return false;
  }
  return true;
}

void
us_msg__msg__StampedArray__fini(us_msg__msg__StampedArray * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // array
  std_msgs__msg__Float32MultiArray__fini(&msg->array);
}

bool
us_msg__msg__StampedArray__are_equal(const us_msg__msg__StampedArray * lhs, const us_msg__msg__StampedArray * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // array
  if (!std_msgs__msg__Float32MultiArray__are_equal(
      &(lhs->array), &(rhs->array)))
  {
    return false;
  }
  return true;
}

bool
us_msg__msg__StampedArray__copy(
  const us_msg__msg__StampedArray * input,
  us_msg__msg__StampedArray * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // array
  if (!std_msgs__msg__Float32MultiArray__copy(
      &(input->array), &(output->array)))
  {
    return false;
  }
  return true;
}

us_msg__msg__StampedArray *
us_msg__msg__StampedArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  us_msg__msg__StampedArray * msg = (us_msg__msg__StampedArray *)allocator.allocate(sizeof(us_msg__msg__StampedArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(us_msg__msg__StampedArray));
  bool success = us_msg__msg__StampedArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
us_msg__msg__StampedArray__destroy(us_msg__msg__StampedArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    us_msg__msg__StampedArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
us_msg__msg__StampedArray__Sequence__init(us_msg__msg__StampedArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  us_msg__msg__StampedArray * data = NULL;

  if (size) {
    data = (us_msg__msg__StampedArray *)allocator.zero_allocate(size, sizeof(us_msg__msg__StampedArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = us_msg__msg__StampedArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        us_msg__msg__StampedArray__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
us_msg__msg__StampedArray__Sequence__fini(us_msg__msg__StampedArray__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      us_msg__msg__StampedArray__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

us_msg__msg__StampedArray__Sequence *
us_msg__msg__StampedArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  us_msg__msg__StampedArray__Sequence * array = (us_msg__msg__StampedArray__Sequence *)allocator.allocate(sizeof(us_msg__msg__StampedArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = us_msg__msg__StampedArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
us_msg__msg__StampedArray__Sequence__destroy(us_msg__msg__StampedArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    us_msg__msg__StampedArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
us_msg__msg__StampedArray__Sequence__are_equal(const us_msg__msg__StampedArray__Sequence * lhs, const us_msg__msg__StampedArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!us_msg__msg__StampedArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
us_msg__msg__StampedArray__Sequence__copy(
  const us_msg__msg__StampedArray__Sequence * input,
  us_msg__msg__StampedArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(us_msg__msg__StampedArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    us_msg__msg__StampedArray * data =
      (us_msg__msg__StampedArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!us_msg__msg__StampedArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          us_msg__msg__StampedArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!us_msg__msg__StampedArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
