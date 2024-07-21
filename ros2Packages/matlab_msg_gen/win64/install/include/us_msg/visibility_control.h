#ifndef US_MSG__VISIBILITY_CONTROL_H_
#define US_MSG__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define US_MSG_EXPORT __attribute__ ((dllexport))
    #define US_MSG_IMPORT __attribute__ ((dllimport))
  #else
    #define US_MSG_EXPORT __declspec(dllexport)
    #define US_MSG_IMPORT __declspec(dllimport)
  #endif
  #ifdef US_MSG_BUILDING_LIBRARY
    #define US_MSG_PUBLIC US_MSG_EXPORT
  #else
    #define US_MSG_PUBLIC US_MSG_IMPORT
  #endif
  #define US_MSG_PUBLIC_TYPE US_MSG_PUBLIC
  #define US_MSG_LOCAL
#else
  #define US_MSG_EXPORT __attribute__ ((visibility("default")))
  #define US_MSG_IMPORT
  #if __GNUC__ >= 4
    #define US_MSG_PUBLIC __attribute__ ((visibility("default")))
    #define US_MSG_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define US_MSG_PUBLIC
    #define US_MSG_LOCAL
  #endif
  #define US_MSG_PUBLIC_TYPE
#endif
#endif  // US_MSG__VISIBILITY_CONTROL_H_
// Generated 21-Jul-2024 17:05:11
 