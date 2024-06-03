#pragma once
#include <iostream>
#include <concepts>
#include <ranges>
#include <assert.h>
#include <immintrin.h>
#include <vector>
#include <bit>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <functional>


using m8=__mmask8;
using m16=__mmask16;
using m32=__mmask32;
using m64=__mmask64;
using m128=__m128i;
using m256=__m256i;
using m512=__m512i;

using f128=__m128;
using f256=__m256;
using f512=__m512;
using d128=__m128d;
using d256=__m256d;
using d512=__m512d;


static_assert(__CHAR_BIT__==8,"Only 8-bit bytes allowed");
static_assert(sizeof(int)==4,"Only 4-byte int allowed");


enum class reg_size
{
    reg8=1,
    reg16=2,
    reg32=4,
    reg64=8,
    reg128=16,
    reg256=32,
    reg512=64
};


template<typename T>
struct reg_type
{};

template<>
struct [[maybe_unused]] reg_type<m8>
{
    using type=m8;
};

template<>
struct [[maybe_unused]] reg_type<m16>
{
    using type=m16;
};

template<>
struct [[maybe_unused]] reg_type<m32>
{
    using type=m32;
};

template<>
struct [[maybe_unused]] reg_type<m64>
{
    using type=m64;
};

template<>
struct [[maybe_unused]] reg_type<m128>
{
    using type=m128;
};

template<>
struct [[maybe_unused]] reg_type<m256>
{
    using type=m256;
};

template<>
struct [[maybe_unused]] reg_type<m512>
{
    using type=m512;
};

template<>
struct [[maybe_unused]] reg_type<f128>
{
    using type=f128;
};

template<>
struct [[maybe_unused]] reg_type<f256>
{
    using type=f256;
};

template<>
struct [[maybe_unused]] reg_type<f512>
{
    using type=f512;
};

template<>
struct [[maybe_unused]] reg_type<d128>
{
    using type=d128;
};


template<>
struct [[maybe_unused]] reg_type<d256>
{
    using type=d256;
};

template<>
struct [[maybe_unused]] reg_type<d512>
{
    using type=d512;
};


template<typename R>
concept is_register=requires()
{
    typename reg_type<std::remove_reference_t<R>>::type;
};


template<typename T>
concept is_register_type=
(
    std::is_integral_v<T> ||
    std::is_floating_point_v<T>
);

template<typename R>
concept is_random_access_storage =requires (R&& r)
{
    requires std::ranges::viewable_range<R>;
    requires std::ranges::random_access_range<R>;
};


template<is_register R,is_register_type T>
constexpr auto set_reg(T value)
{
    using type   = typename reg_type<R>::type;
    using v_type = std::remove_reference_t<T>;

    if constexpr(std::is_integral_v<v_type>)
    {
        if constexpr((std::is_same_v<type,m128>) && (sizeof(value)==sizeof(m8)))
            return _mm_set1_epi8(value);
        
        else if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m16)))
            return _mm_set1_epi16(value);

        else  if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m32)))
            return _mm_set1_epi32(value);

        else if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m64)))
            return _mm_set1_epi64(value);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m8)))
            return _mm256_set1_epi8(value);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m16)))
            return _mm256_set1_epi16(value);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m32)))
            return _mm256_set1_epi32(value);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m64)))
            return _mm256_set1_epi64(value);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m128)))
            return _mm256_set_m128i(value,value);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m8)))
            return _mm512_set1_epi8(value);
        
        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m16)))
            return _mm512_set1_epi16(value);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m32)))
            return _mm512_set1_epi32(value);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m64)))
            return _mm512_set1_epi64(value);

        else 
            return;
    }
    else if(std::is_floating_point_v<v_type>)
    {
        if constexpr((std::is_same_v<type,f128>)&&(std::is_same_v<v_type,float>))
            return _mm_set1_ps(value);

        if constexpr((std::is_same_v<type,f256>)&&(std::is_same_v<v_type,float>))
            return _mm256_set1_ps(value);

        if constexpr((std::is_same_v<type,f512>)&&(std::is_same_v<v_type,float>))
            return _mm512_set1_ps(value);

        if constexpr((std::is_same_v<type,d128>)&&(std::is_same_v<v_type,double>))
            return _mm_set1_pd(value);

        if constexpr((std::is_same_v<type,d256>)&&(std::is_same_v<v_type,double>))
            return _mm256_set1_pd(value);

         if constexpr((std::is_same_v<type,d512>)&&(std::is_same_v<v_type,double>))
            return _mm512_set1_pd(value);       
    }

}



template<is_register R,is_register_type T,std::same_as<T>...Types>
constexpr auto set_regs(T value,Types...values)
{
    using type   = typename reg_type<R>::type;
    using v_type = std::remove_reference_t<T>;

    constexpr size_t number_values=sizeof...(Types)+1;

    if constexpr(std::is_integral_v<v_type>)
    {
        if constexpr((std::is_same_v<type,m128>) && (sizeof(value)==sizeof(m8)))
            return _mm_set_epi8(value,values...);
        
        else if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m16)))
            return _mm_set_epi16(value,values...);

        else  if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m32)))
            return _mm_set_epi32(value,values...);

        else if constexpr((std::is_same_v<type,m128>)&&(sizeof(value)==sizeof(m64)))
            return _mm_set_epi64(value,values...);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m8)))
            return _mm256_set_epi8(value,values...);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m16)))
            return _mm256_set_epi16(value,values...);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m32)))
            return _mm256_set_epi32(value,values...);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m64)))
            return _mm256_set_epi64(value,values...);

        else if constexpr((std::is_same_v<type,m256>)&&(sizeof(value)==sizeof(m128)))
            return _mm256_set_m128i(value,values...);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m8)))
            return _mm512_set_epi8(value,values...);
        
        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m16)))
            return _mm512_set_epi16(value,values...);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m32)))
            return _mm512_set_epi32(value,values...);

        else if constexpr((std::is_same_v<type,m512>)&&(sizeof(value)==sizeof(m64)))
            return _mm512_set_epi64(value,values...);

        else 
            return;
    }
    else if(std::is_floating_point_v<v_type>)
    {
        if constexpr((std::is_same_v<type,f128>)&&(std::is_same_v<v_type,float>))
            return _mm_set_ps(value,values...);

        if constexpr((std::is_same_v<type,f256>)&&(std::is_same_v<v_type,float>))
            return _mm256_set_ps(value,values...);

        if constexpr((std::is_same_v<type,f512>)&&(std::is_same_v<v_type,float>))
            return _mm512_set_ps(value,values...);

        if constexpr((std::is_same_v<type,d128>)&&(std::is_same_v<v_type,double>))
            return _mm_set_pd(value,values...);

        if constexpr((std::is_same_v<type,d256>)&&(std::is_same_v<v_type,double>))
            return _mm256_set_pd(value,values...);

         if constexpr((std::is_same_v<type,d512>)&&(std::is_same_v<v_type,double>))
            return _mm512_set_pd(value,values...);       
    }

}


template<is_register Reg,std::ranges::viewable_range R>
requires requires(R&& r)
{
    requires (
        is_register_type<std::remove_reference_t<decltype(*r)>> 
        ||
        is_register_type<std::remove_reference_t<decltype(*r.begin())>>
    );
}
constexpr auto set_regs(R&& r)
{

    std::ranges::ref_view range=r;
    using type=std::remove_reference_t<decltype(*range.begin())>;

    if((r.size()*sizeof(type)<sizeof(Reg)) || ((sizeof(Reg)%r.size())&1))
        throw std::logic_error{"size values into contaiter doesn't match the register type"};

    size_t blocks=range.size()/sizeof(type);

    type arr[range.size()];

    return 0;
}


template<is_random_access_storage T,is_register R>
constexpr void storeu_reg(T&& arr,R&& reg)
{
    using type=typename reg_type<std::remove_reference_t<R>>::type;
    std::ranges::ref_view view=arr;

    using v_type=std::remove_reference_t<decltype(*arr)>;

    if constexpr((std::is_same_v<type,m128>) && (sizeof(v_type)==sizeof(m8)))
       { _mm_storeu_si128((m128*)arr,reg); return;}

    else if constexpr((std::is_same_v<type,m128>) && (sizeof(v_type)==sizeof(m16)))
       { _mm_storeu_si128((m128*)arr,reg); return;}

    else if constexpr ((std::is_same_v<type,m128>) && (sizeof(v_type)==sizeof(m32)))
       { _mm_storeu_si128((m128*)arr,reg); return; } 

    else if constexpr ((std::is_same_v<type,m128>) && (sizeof(v_type)==sizeof(m64)))
        {_mm_storeu_si128((m128*)arr,reg); return; }

    else if  constexpr  ((std::is_same_v<type,m128>) && (sizeof(v_type)==sizeof(m128)))
        {_mm_storeu_si128((m128*)arr,reg); return; }

    else if constexpr  ((std::is_same_v<type,m256>) && (sizeof(v_type)==sizeof(m8)))
        {_mm256_storeu_si256((m256*)arr,reg); return; }

    else if constexpr  ((std::is_same_v<type,m256>) && (sizeof(v_type)==sizeof(m16)))
        {_mm256_storeu_si256((m256*)arr,reg); return; }

    else if  constexpr  ((std::is_same_v<type,m256>) && (sizeof(v_type)==sizeof(m32)))
        {_mm256_storeu_si256(reinterpret_cast<m256*>(arr),reg); return;}

    else if constexpr ((std::is_same_v<type,m256>) && (sizeof(v_type)==sizeof(m64)))
        {_mm256_storeu_si256(reinterpret_cast<m256*>(arr),reg); return;}

    else if constexpr ((std::is_same_v<type,m256>) && (sizeof(v_type)==sizeof(m256)))
        {_mm256_storeu_si256(reinterpret_cast<m256*>(arr),reg); return;}

    else if constexpr ((std::is_same_v<type,m512>) && (sizeof(v_type)==sizeof(m8)))
        {_mm512_storeu_epi8(arr,reg); return;}

    else if constexpr  ((std::is_same_v<type,m512>) && (sizeof(v_type)==sizeof(m16)))
        {_mm512_storeu_epi16(arr,reg); return;}

    else  if constexpr ((std::is_same_v<type,m512>) && (sizeof(v_type)==sizeof(m32)))
        {_mm512_storeu_epi32(arr,reg); return;}

    else if constexpr ((std::is_same_v<type,m512>) && (sizeof(v_type)==sizeof(m64)))
        {_mm512_storeu_epi64(arr,reg); return;}

   else if  constexpr  ((std::is_same_v<type,m512>) && (sizeof(v_type)==sizeof(m512)))
        {_mm512_storeu_si512(arr,reg); return;}

   else if constexpr ((std::is_same_v<type,f128>) && (std::is_same_v<v_type,float>))
        {_mm_storeu_ps(arr,reg); return;}

    else if  constexpr ((std::is_same_v<type,f256>) && (std::is_same_v<v_type,float>))
        {_mm256_storeu_ps(arr,reg); return;}

    else if constexpr ((std::is_same_v<type,f512>) && (std::is_same_v<v_type,float>))
        {_mm512_storeu_ps(arr,reg); return;}

    else if constexpr ((std::is_same_v<type,d128>) && (std::is_same_v<v_type,double>))
        {_mm_storeu_pd(arr,reg); return;} 

    else if constexpr ((std::is_same_v<type,d256>) && (std::is_same_v<v_type,double>))
        {_mm256_storeu_pd(arr,reg); return;}

    else if constexpr ((std::is_same_v<type,d512>) && (std::is_same_v<v_type,double>))
        {_mm512_storeu_pd(arr,reg); return;} 
   else
     throw std::logic_error{"Function not found"};
     
}




template<is_register_type T,is_register R>
void print_register(R&& r)
{
    constexpr size_t sz=sizeof(R)/sizeof(T);
    T array[sz]{};
    storeu_reg(array,std::forward<R>(r));

   for(size_t i=sz-1;i!=0;i--)
   {
      std::cout<<array[i]<<"      ";
   }
      std::cout<<array[0]<<"      "<<"\n";   
}
    

bool equal_xmm(m128& lhs,m128& rhs)
{
    m128 mask=_mm_cmpeq_epi32(lhs,rhs);
    m64 value=_mm_extract_epi64(mask,0);
    value&=_mm_extract_epi64(mask,1);

    return (~value==0);
}


bool equal_ymm(m256& lhs, m256& rhs)
{
    m128 lhs_low = _mm256_extractf128_si256(lhs,0);
    m128 rhs_low = _mm256_extractf128_si256(rhs,0);

    m128 lhs_high = _mm256_extractf128_si256(lhs,1);
    m128 rhs_high = _mm256_extractf128_si256(rhs,1);

    return (equal_xmm(lhs_low,rhs_low) & equal_xmm(lhs_high,rhs_high));
}


int bitscan_not_equal_xmm(m128& lhs,m128& rhs)
{
    if(equal_xmm(lhs,rhs))
        return -1;

    m128 mask=_mm_cmpeq_epi32(lhs,rhs);
    m64 high=_mm_extract_epi64(mask,1);
    m64 low = _mm_extract_epi64(mask,0);

    if(~low==0)
    {
        return __builtin_ffsll(~high)/32+2;
    }
    return __builtin_ffsll(~low)/32;
}


int bitscan_not_equal_ymm(m256& lhs,m256& rhs)
{
    if(equal_ymm(lhs,rhs))
        return -1;

    m128 left_low  = _mm256_extractf128_si256(lhs,0);
    m128 right_low = _mm256_extractf128_si256(rhs,0);

    m128 left_high  = _mm256_extractf128_si256(lhs,1);
    m128 right_high = _mm256_extractf128_si256(rhs,1);

    int pos_l=bitscan_not_equal_xmm(left_low,right_low);
    int pos_h=bitscan_not_equal_xmm(left_high,right_high);

    return (pos_l>=pos_h) ? pos_l : pos_h+4;
}



int bitscan_equal_xmm(m128& lhs,m128& rhs)
{
    if(equal_xmm(lhs,rhs))
        return 0;

    m128 mask=_mm_cmpeq_epi32(lhs,rhs);
    m64 high=_mm_extract_epi64(mask,1);
    m64 low = _mm_extract_epi64(mask,0);

    if(low==0)
    {
        return __builtin_ffsll(high)/32+2;
    }
    return __builtin_ffsll(low)/32;
}


int bitscan_equal_ymm(m256& lhs,m256& rhs)
{
    if(equal_ymm(lhs,rhs))
        return 0;

    m128 left_low  = _mm256_extractf128_si256(lhs,0);
    m128 right_low = _mm256_extractf128_si256(rhs,0);

    m128 left_high  = _mm256_extractf128_si256(lhs,1);
    m128 right_high = _mm256_extractf128_si256(rhs,1);

    int pos_l=bitscan_equal_xmm(left_low,right_low);
    int pos_h=bitscan_equal_xmm(left_high,right_high);

    return (pos_l>=pos_h) ? pos_l : pos_h+4;
}


template<std::ranges::viewable_range R,typename T>
requires requires (R&& r,T&& value)
{
    std::is_integral_v<std::remove_reference_t<T>>;
    {*r}->std::same_as<T&>;
    requires std::ranges::random_access_range<R>;
}
int find(R&& r,T&& value)
{
    std::ranges::ref_view range=r;
    size_t sz=(range.size()/8)*8;

    m256 f=set_ymm(value);

    for(size_t i=0;i<sz;i+=8)
    {
        m256 reg = _mm256_loadu_si256((m256*)(range.data()+i));

        int pos=bitscan_equal_ymm(reg,f);
        if(pos!=-1)
            return pos+i;         
    }

    for(auto i=range.begin()+sz;i!=range.end();i++) 
        if(*i==value)
            return (i-range.begin());

    return -1; 
}



