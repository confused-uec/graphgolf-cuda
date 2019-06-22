#pragma once
#include <array>
namespace graphgolf{
    template<typename elem, int capacity> class arrayQueue{
    private:
        std::array<int,capacity> arr;
        unsigned int f=0,b=0;
        int count=0;
    public:
        arrayQueue(){
            f=b=count=0;
        }
        inline void clear(){
            f=b=count=0;
        } 
        inline bool empty(){
            return count==0;
        }
        inline void push(elem value){
            arr[b]=value;
            if(++b==capacity) b=0;
            count++;
        }
        inline elem front(){
            return arr[f];
        }
        inline void pop(){
            if(++f==capacity) f=0;
            count--;
        }
        inline int size(){
            return count;
        }
    };
}