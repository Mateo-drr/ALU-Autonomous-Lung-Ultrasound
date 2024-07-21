// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from us_msg:msg\StampedArray.idl
// generated code does not contain a copyright notice

#ifndef US_MSG__MSG__DETAIL__STAMPED_ARRAY__BUILDER_HPP_
#define US_MSG__MSG__DETAIL__STAMPED_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "us_msg/msg/detail/stamped_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace us_msg
{

namespace msg
{

namespace builder
{

class Init_StampedArray_array
{
public:
  explicit Init_StampedArray_array(::us_msg::msg::StampedArray & msg)
  : msg_(msg)
  {}
  ::us_msg::msg::StampedArray array(::us_msg::msg::StampedArray::_array_type arg)
  {
    msg_.array = std::move(arg);
    return std::move(msg_);
  }

private:
  ::us_msg::msg::StampedArray msg_;
};

class Init_StampedArray_stamp
{
public:
  Init_StampedArray_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_StampedArray_array stamp(::us_msg::msg::StampedArray::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_StampedArray_array(msg_);
  }

private:
  ::us_msg::msg::StampedArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::us_msg::msg::StampedArray>()
{
  return us_msg::msg::builder::Init_StampedArray_stamp();
}

}  // namespace us_msg

#endif  // US_MSG__MSG__DETAIL__STAMPED_ARRAY__BUILDER_HPP_
