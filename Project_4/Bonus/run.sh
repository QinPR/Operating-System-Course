#!/bin/bash
nvcc --relocatable-device-code=true main.cu file_system.cu user_program.cu -o main.out