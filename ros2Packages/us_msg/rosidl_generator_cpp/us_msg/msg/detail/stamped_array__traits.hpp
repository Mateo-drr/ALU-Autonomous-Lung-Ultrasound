// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from us_msg:msg/StampedArray.idl
// generated code does not contain a copyright notice

#ifndef US_MSG__MSG__DETAIL__STAMPED_ARRAY__TRAITS_HPP_
#define US_MSG__MSG__DETAIL__STAMPED_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "us_msg/msg/detail/stamped_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'array'
#include "std_msgs/msg/detail/float32_multi_array__traits.hpp"

namespace us_msg
{

namespace msg
{

inline void to_flow_style_yaml(
  const StampedArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: array
  {
    out << "array: ";
    to_flow_style_yaml(msg.array, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const StampedArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: array
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "array:\n";
    to_block_style_yaml(msg.array, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const StampedArray & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace us_msg

namespace rosidl_generator_traits
{

[[deprecated("use us_msg::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const us_msg::msg::StampedArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  us_msg::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use us_msg::msg::to_yaml() instead")]]
inline std::string to_yaml(const us_msg::msg::StampedArray & msg)
{
  return us_msg::msg::to_yaml(msg);
}

template<>
inline const char * data_type<us_msg::msg::StampedArray>()
{
  return "us_msg::msg::StampedArray";
}

template<>
inline const char * name<us_msg::msg::StampedArray>()
{
  return "us_msg/msg/StampedArray";
}

template<>
struct has_fixed_size<us_msg::msg::StampedArray>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value && has_fixed_size<std_msgs::msg::Float32MultiArray>::value> {};

template<>
struct has_bounded_size<us_msg::msg::StampedArray>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value && has_bounded_size<std_msgs::msg::Float32MultiArray>::value> {};

template<>
struct is_message<us_msg::msg::StampedArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // US_MSG__MSG__DETAIL__STAMPED_ARRAY__TRAITS_HPP_
