#include "intrinsic_head.hpp"


void blend()
{
    auto z0=set_reg<m128>((m16)10);
    auto z1=set_reg<m128>((m16)0);

    auto vec=_mm_blend_epi16(z1,z0,0xA);

    print_register<m16>(vec);   
}


auto get_mask(m256 vec)
{
    int arr[8]{};
    storeu_reg(arr,vec);
    int mask=0;

    for(size_t i=0;i<8;i++)
    {
        mask|=arr[i]<<i;
    }
    return mask;
}

template<is_random_access_storage R>
auto findmin_simd(R&& r)
{
    std::ranges::ref_view range=r;

    int arr[8];

    auto min_el=INT32_MAX;
    auto min = set_reg<m256>(INT32_MAX);

    int mainsz=(range.size()/8)*8;
    

    for(int i=0;i<mainsz;i+=8)
    {
         auto args = _mm256_loadu_si256((m256*) (range.data() + i));
         min = _mm256_min_epi32(args,min);   
    }

    storeu_reg(arr,min);

    for(int i=0;i<8;i++)
    {
        if(min_el>arr[i])
        {
            min_el=arr[i];
        }
    }

    min=set_reg<m256>(min_el);

    //now we will find an index that min element

    int index=-1;
    for(int i=0;i<mainsz;i+=8)
    {
        auto args=_mm256_loadu_si256((m256*) (range.data() + i));
        auto mask=_mm256_cmpeq_epi32(args,min);
        size_t zeromask= _mm256_movemask_epi8(mask);
        if(zeromask!=0)
        {
            index=(__builtin_ffs(zeromask)/sizeof(int))+i; 
            break;
        }
    }

    //handle the tail

    for(int i=mainsz;i<range.size();i++)
    {
        if(min_el>range[i])
        {
            min_el=range[i];
            index=i;
        }
    }

    return index;
}


template<typename R>
void print(R&& r)
{
    std::ranges::ref_view ran=r;

    for(auto i=ran.begin();i!=ran.end();i++)
    {
        std::cout<<*i<<"\n";
    }
}


template<typename T>
auto simple_find(T&& t)
{
    auto it=std::min_element(t.begin(),t.end());

    return it-t.begin();
}


template<typename F,typename...Args>
auto time(F&& f,Args&&...args)
{
    const auto start{std::chrono::high_resolution_clock::now()};
    std::invoke(std::forward<F>(f),std::forward<Args>(args)...);
    const auto end{std::chrono::high_resolution_clock::now()};

   auto elapsed_time= std::chrono::duration_cast<std::chrono::milliseconds> (end-start);

   return elapsed_time;
}

int main()
{
     size_t sz=1'000'000'0;

     std::vector<int> v(sz);

     std::random_device rd;
     std::mt19937 g(rd());

     for(auto &i:v)
     {
        i=g();
     }


    std::shuffle(v.begin(),v.end(),g);


    std::cout<<"simd find time:"<<time(findmin_simd<std::vector<int>&>,v)<<"\n";

    std::cout<<"simple_find time:"<<time(simple_find<std::vector<int>&>,v)<<"\n";


    return 0;
}
