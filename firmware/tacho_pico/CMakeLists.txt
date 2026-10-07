cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2 CACHE STRING "Board type")
include(pico_sdk_import.cmake)

project(tacho_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

execute_process(
    COMMAND git describe --always --dirty --abbrev=8
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE FW_GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT FW_GIT_VERSION)
    set(FW_GIT_VERSION "nogit")
endif()

function(tacho_target name selftest)
    add_executable(${name} src/main.c src/codec.c)
    pico_generate_pio_header(${name} ${CMAKE_CURRENT_LIST_DIR}/src/capture.pio)
    target_include_directories(${name} PRIVATE src)
    target_compile_definitions(${name} PRIVATE
        TACHO_SELFTEST=${selftest}
        FW_VERSION="${FW_GIT_VERSION}"
        PICO_STDIO_USB_RESET_INTERFACE_SUPPORT_RESET_TO_BOOTSEL=1)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${name} pico_stdlib hardware_pio hardware_dma hardware_uart
        hardware_pwm hardware_timer hardware_watchdog hardware_clocks)
    # USB only for picotool's reset interface; the link is UART0.
    pico_enable_stdio_usb(${name} 1)
    pico_enable_stdio_uart(${name} 0)
    pico_add_extra_outputs(${name})
endfunction()

tacho_target(tacho_pico 0)
tacho_target(tacho_pico_selftest 1)
