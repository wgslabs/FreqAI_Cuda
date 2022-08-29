#pragma once
#include <iostream>
#include <chrono>
#include <string>
#include <memory>

namespace WGSTest
{
	using namespace std;
	using namespace chrono;
	class Timer
	{
	public:
		Timer()
		{
			timer = system_clock::now();
			Reset();
		}
		~Timer() { Reset(); }

		void Start() { timer = system_clock::now(); }
		void End()
        {
			Reset();
            duration_ptr = make_unique<duration<double>>(system_clock::now() - timer);
        }
		void Reset() { duration_ptr.reset(); }

		void Print() { cout << duration_ptr->count() << " sec" << endl; }
		void Print(string task_name)
        {
            cout << task_name << " / ";
            Print();
        }
	private:
		system_clock::time_point timer;
		unique_ptr<duration<double>> duration_ptr;
	};
}