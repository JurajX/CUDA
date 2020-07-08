#include <iostream>
#include <string>
#include <filesystem>

#include "HW.hpp"


int main(int argc, char **argv) {
    std::cout << "Hello, World! Welcom to " << std::string(argv[0]) << ".\n";

    std::cout << "Please enter the homework number you would like to execute." << '\n'
              << "Possible options are: 1, 2, 3, 4. To terminate enter any letter." << '\n'
              << "Your input: ";

    std::filesystem::path cwd = std::filesystem::current_path();    // build path
    cwd = cwd.parent_path()/"images";                               // path to images

    int i;
    while (std::cin >> i) {
        switch (i) {
            case 1:
                std::cout << "\n======= HOMEWORK 1 =======" << '\n';
                HW1(cwd, "HW1.jpg");
                break;
            case 2:
                std::cout << "\n======= HOMEWORK 2 =======" << '\n';
                HW2(cwd, "HW2.jpg");
                break;
            case 3:
                std::cout << "\n======= HOMEWORK 3 =======" << '\n';
                HW3(cwd, "HW3.png");
                break;
            case 4:
                std::cout << "\n======= HOMEWORK 4 =======" << '\n';
                HW4(cwd, "HW4.jpg", "HW4_red_eye.jpg");
                break;
            default:
                std::cout << "Invalid homework number " << i << "." << '\n';
        }
        std::cout << '\n' << "If you would like to continue, enter another homework number." << '\n'
                  << "Your input: ";
    }

    std::cout << "Program is terminating. Thank you for using " << std::string(argv[0]) << std::endl;
    return 0;
}
