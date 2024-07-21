// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice

#ifndef US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_HPP_
#define US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'array'
#include "std_msgs/msg/detail/float32_multi_array__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__us_msg__msg__StampedArray __attribute__((deprecated))
#else
# define DEPRECATED__us_msg__msg__StampedArray __declspec(deprecated)
#endif

namespace us_msg
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct StampedArray_
{
  using Type = StampedArray_<ContainerAllocator>;

  explicit StampedArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    array(_init)
  {
    (void)_init;
  }

  explicit StampedArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    array(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _array_type =
    std_msgs::msg::Float32MultiArray_<ContainerAllocator>;
  _array_type array;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__array(
    const std_msgs::msg::Float32MultiArray_<ContainerAllocator> & _arg)
  {
    this->array = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    us_msg::msg::StampedArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const us_msg::msg::StampedArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<us_msg::msg::StampedArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<us_msg::msg::StampedArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      us_msg::msg::StampedArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<us_msg::msg::StampedArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      us_msg::msg::StampedArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<us_msg::msg::StampedArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<us_msg::msg::StampedArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<us_msg::msg::StampedArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__us_msg__msg__StampedArray
    std::shared_ptr<us_msg::msg::StampedArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__us_msg__msg__StampedArray
    std::shared_ptr<us_msg::msg::StampedArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const StampedArray_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->array != other.array) {
      return false;
    }
    return true;
  }
  bool operator!=(const StampedArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct StampedArray_

// alias to use template instance with default allocator
using StampedArray =
  us_msg::msg::StampedArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace us_msg

#endif  // US_MSG__MSG__DETAIL__STAMPED_ARRAY__STRUCT_HPP_
