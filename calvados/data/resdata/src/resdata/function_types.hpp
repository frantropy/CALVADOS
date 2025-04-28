#ifndef _RESDATA_FUNCTION_TYPES_HPP
#define _RESDATA_FUNCTION_TYPES_HPP

#include <gromacs/pbcutil/pbc.h>

#include <unordered_map>

namespace resdata::ftypes
{
  template<typename T>
  struct function_traits;
  
  template<typename Ret, typename... Args>
  struct function_traits<Ret(*)(Args...)> {
      using return_type = Ret;
      using args_tuple = std::tuple<Args...>;
      // Define a type for the function signature
      using signature = Ret(Args...);
  };
  
  // does nothing while taking the same arguments as the function
  template<typename function_traits>
  auto do_nothing() {
      return [](auto&&... args) -> typename function_traits::return_type {};
  }
}

namespace resdata::dtypes
{
  static inline std::unordered_map<std::string, PbcType> pbc_type_map = {
    {"xyz", PbcType::Xyz},
    {"no", PbcType::No},
    {"xy", PbcType::XY},
    {"screw", PbcType::Screw},
    {"unset", PbcType::Unset}
  };
}


#endif // _RESDATA_FUNCTION_TYPES_HPP