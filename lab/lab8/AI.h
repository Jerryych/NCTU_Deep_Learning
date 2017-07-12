#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include "2048.h"

class experience {
public:
    state sp;
    state spp;
};


class AI {
public:
    static void load_tuple_weights() {
        //std::string filename = "ori.weight";                   // put the name of weight file here
        std::string filename = "self.weight";
	std::ifstream in;
        in.open(filename.c_str(), std::ios::in | std::ios::binary);
        if (in.is_open()) {
            for (size_t i = 0; i < feature::list().size(); i++) {
                in >> *(feature::list()[i]);
                std::cout << feature::list()[i]->name() << " is loaded from " << filename << std::endl;
            }
            in.close();
        }
    }

    static void set_tuples() {
	//standard
	/*feature::list().push_back(new pattern<4>(0, 4, 8, 12));
	feature::list().push_back(new pattern<4>(1, 5, 9, 13));
	feature::list().push_back(new pattern<6>(0, 4, 8, 12, 1, 5));
	feature::list().push_back(new pattern<6>(1, 5, 9, 13, 2, 6));*/
	//self
	feature::list().push_back(new pattern<4>(0, 4, 8, 12));
	feature::list().push_back(new pattern<4>(1, 5, 9, 13));
	feature::list().push_back(new pattern<6>(0, 1, 2, 3, 4, 5));
	feature::list().push_back(new pattern<6>(4, 5, 6, 7, 8, 9));
    }

    static int get_best_move(state s) {         // return best move dir
	int r = 0, argmax = 0;
	float v = 0.0, max = -32767.0, temp = 0.0;
	for (int i = 0; i < 4; i++) {
		state t(s);
		r = t.move(i);
		v = t.evaluate_score();
		temp = r + v;
		if (max < temp && r >= 0) {
			max = temp;
			argmax = i;
		}
	}
	return argmax;
    }

};
