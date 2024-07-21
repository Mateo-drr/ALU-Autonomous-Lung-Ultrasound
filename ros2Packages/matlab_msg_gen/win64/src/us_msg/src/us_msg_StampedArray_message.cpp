// Copyright 2020-2022 The MathWorks, Inc.
// Common copy functions for us_msg/StampedArray
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4265)
#pragma warning(disable : 4456)
#pragma warning(disable : 4458)
#pragma warning(disable : 4946)
#pragma warning(disable : 4244)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif //_MSC_VER
#include "rclcpp/rclcpp.hpp"
#include "us_msg/msg/stamped_array.hpp"
#include "visibility_control.h"
#include "class_loader/multi_library_class_loader.hpp"
#include "ROS2PubSubTemplates.hpp"
class US_MSG_EXPORT ros2_us_msg_msg_StampedArray_common : public MATLABROS2MsgInterface<us_msg::msg::StampedArray> {
  public:
    virtual ~ros2_us_msg_msg_StampedArray_common(){}
    virtual void copy_from_struct(us_msg::msg::StampedArray* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const us_msg::msg::StampedArray* msg, MultiLibLoader loader, size_t size = 1);
};
  void ros2_us_msg_msg_StampedArray_common::copy_from_struct(us_msg::msg::StampedArray* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //stamp
        const matlab::data::StructArray stamp_arr = arr["stamp"];
        auto msgClassPtr_stamp = getCommonObject<builtin_interfaces::msg::Time>("ros2_builtin_interfaces_msg_Time_common",loader);
        msgClassPtr_stamp->copy_from_struct(&msg->stamp,stamp_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'stamp' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'stamp' is wrong type; expected a struct.");
    }
    try {
        //array
        const matlab::data::StructArray array_arr = arr["array"];
        auto msgClassPtr_array = getCommonObject<std_msgs::msg::Float32MultiArray>("ros2_std_msgs_msg_Float32MultiArray_common",loader);
        msgClassPtr_array->copy_from_struct(&msg->array,array_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'array' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'array' is wrong type; expected a struct.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T ros2_us_msg_msg_StampedArray_common::get_arr(MDFactory_T& factory, const us_msg::msg::StampedArray* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","stamp","array"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("us_msg/StampedArray");
    // stamp
    auto currentElement_stamp = (msg + ctr)->stamp;
    auto msgClassPtr_stamp = getCommonObject<builtin_interfaces::msg::Time>("ros2_builtin_interfaces_msg_Time_common",loader);
    outArray[ctr]["stamp"] = msgClassPtr_stamp->get_arr(factory, &currentElement_stamp, loader);
    // array
    auto currentElement_array = (msg + ctr)->array;
    auto msgClassPtr_array = getCommonObject<std_msgs::msg::Float32MultiArray>("ros2_std_msgs_msg_Float32MultiArray_common",loader);
    outArray[ctr]["array"] = msgClassPtr_array->get_arr(factory, &currentElement_array, loader);
    }
    return std::move(outArray);
  } 
class US_MSG_EXPORT ros2_us_msg_StampedArray_message : public ROS2MsgElementInterfaceFactory {
  public:
    virtual ~ros2_us_msg_StampedArray_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType /*type*/);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType /*type*/);
    virtual std::shared_ptr<void> generateCppMessage(ElementType /*type*/, const matlab::data::StructArray& /* arr */, MultiLibLoader /* loader */, std::map<std::string,std::shared_ptr<MATLABROS2MsgInterfaceBase>>* /*commonObjMap*/);
    virtual matlab::data::StructArray generateMLMessage(ElementType  /*type*/ ,void*  /* msg */, MultiLibLoader /* loader */ , std::map<std::string,std::shared_ptr<MATLABROS2MsgInterfaceBase>>* /*commonObjMap*/);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          ros2_us_msg_StampedArray_message::generatePublisherInterface(ElementType /*type*/){
    return std::make_shared<ROS2PublisherImpl<us_msg::msg::StampedArray,ros2_us_msg_msg_StampedArray_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         ros2_us_msg_StampedArray_message::generateSubscriberInterface(ElementType /*type*/){
    return std::make_shared<ROS2SubscriberImpl<us_msg::msg::StampedArray,ros2_us_msg_msg_StampedArray_common>>();
  }
  std::shared_ptr<void> ros2_us_msg_StampedArray_message::generateCppMessage(ElementType /*type*/, 
                                           const matlab::data::StructArray& arr,
                                           MultiLibLoader loader,
                                           std::map<std::string,std::shared_ptr<MATLABROS2MsgInterfaceBase>>* commonObjMap){
    auto msg = std::make_shared<us_msg::msg::StampedArray>();
    ros2_us_msg_msg_StampedArray_common commonObj;
    commonObj.mCommonObjMap = commonObjMap;
    commonObj.copy_from_struct(msg.get(), arr[0], loader);
    return msg;
  }
  matlab::data::StructArray ros2_us_msg_StampedArray_message::generateMLMessage(ElementType  /*type*/ ,
                                                    void*  msg ,
                                                    MultiLibLoader  loader ,
                                                    std::map<std::string,std::shared_ptr<MATLABROS2MsgInterfaceBase>>*  commonObjMap ){
    ros2_us_msg_msg_StampedArray_common commonObj;	
    commonObj.mCommonObjMap = commonObjMap;	
    MDFactory_T factory;
    return commonObj.get_arr(factory, (us_msg::msg::StampedArray*)msg, loader);			
 }
#include "class_loader/register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(ros2_us_msg_msg_StampedArray_common, MATLABROS2MsgInterface<us_msg::msg::StampedArray>)
CLASS_LOADER_REGISTER_CLASS(ros2_us_msg_StampedArray_message, ROS2MsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER