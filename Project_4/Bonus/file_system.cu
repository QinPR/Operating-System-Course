#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

#define DCB_SIZE 212    /* the smallest size of block for directory control */

__device__ __managed__ u32 gtime = 0;

__device__ void init_FBC(FileSystem *fs){
  for (int base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    fs->volume[base_addr_of_FCB+fs->FCB_SIZE-1] == 0;
  }
}

__device__ void init_DCB(FileSystem *fs){
  for (int i = 0; i < DCB_SIZE * fs->MAX_FILE_NUM; i++){
    *(uchar*)&fs->directory_control_memory[i] = 0;
  }
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, uchar * directory_control_memory, int current_dir)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;  /* contians volume details. # of free blocks, block size, free block pointers to the array */
                                          /* the structure of super block is: 32*1024bit = 4096bytes, each bit represents the state of block*/
  fs->FCB_SIZE = FCB_SIZE;                /* store information about a file */
                                          /* The structure of FCB: |20B: name|2B: size|4B: time|4B: pointer|1B:NULL|1B:valid| */
  fs->FCB_ENTRIES = FCB_ENTRIES;          /* total number of blocks = 32*1024B */
  fs->STORAGE_SIZE = VOLUME_SIZE;         /* total size of the volume = 1085440B = 4096(super block)+32768(FCB)+1048576(Contents of files)*/
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;  /* How large one block is = 32B */
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;    /* The maximal size of filename = 20B */
  fs->MAX_FILE_NUM = MAX_FILE_NUM;              /* Maximal number of files = 1024 */
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;            /* The maximal size of filename = 1024B */
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;    /* The base addr of contents of files = 36864 = 4096(super)+32768(FCB) */

  fs->directory_control_memory = directory_control_memory;
  fs->current_dir = current_dir;
  init_DCB(fs);
  *(uchar*)&fs->directory_control_memory[DCB_SIZE-1] = 1;

  init_FBC(fs);   /* set all the valid bit in FBC to 0 */
}

/* this function is used to find an empty FCB */
__device__ u32 find_free_FBC(FileSystem *fs){
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    if (fs->volume[base_addr_of_FCB+fs->FCB_SIZE-1] == 0) return base_addr_of_FCB;
  }
  printf("FCB has all been used!\n");
  return -1;
}

/* this function would check whether two files has same name */
__device__ int check_name(char* a, uchar*b){
  int i = 0;
  while (a[i] != '\0'){
    if (a[i] != b[i]) return 0;     /* they got different name */
    i += 1;
  }
  if (b[i] != '\0') return 0;
  return 1;                         /* same name */
}

__device__ int check_pointer(FileSystem *fs, u32 exit_pointer, u32 iter_pointer){

  if (exit_pointer < *(u32*)&fs->volume[iter_pointer]) return 1;
  else{
    return 0;
  } 
}

__device__ int check_total_files(FileSystem *fs){
  int total = 0;
  for (int base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE + DCB_SIZE - 4; base_addr_of_FCB+=4){
    if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] >= 1){
      total += 1;
    }
  }
  return total;
}

__device__ void list_by_time(FileSystem *fs){
  int latest_time = -1;
  u32 global_latest_time = 0xFFFFFFFF;
  u32 latest_FCB = 0;
  int file_numbers = check_total_files(fs);
  for (int i = 0; i < file_numbers; i++){
    for (int base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE + DCB_SIZE - 4; base_addr_of_FCB+=4){
      if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] >= 1){
        if (*(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 22] >= latest_time && *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 22] < global_latest_time){
          latest_time = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 22];
          latest_FCB = *(int*)&fs->directory_control_memory[base_addr_of_FCB];
        }
      }
    }
    global_latest_time = latest_time;
    latest_time = -1;
    int j = 0;
    do{
      printf("%c", fs->volume[latest_FCB+j]);
      j += 1;
    }while (fs->volume[latest_FCB+j] != '\0');
    if (fs->volume[latest_FCB + fs->FCB_SIZE-1] == 3){    /* when it is a directory */
      printf(" d");
    }
    printf("\n");
  }
}

__device__ void list_by_size(FileSystem *fs){
  int biggest_size = 0;
  int global_biggest_size = 0x7FFF;
  int biggest_FCB = 0;
  int latest_time = 0x7FFF;
  int global_latest_time = -1;
  int file_numbers = check_total_files(fs);
  /* if the sizes are the same, sort by the create time */
  for (int i = 0; i < file_numbers; i++){
    for (int base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE + DCB_SIZE - 4; base_addr_of_FCB+=4){
      if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] >= 1){
        if (*(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28] >= biggest_size && *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28] < global_biggest_size){
          biggest_size = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28];
          biggest_FCB = *(int*)&fs->directory_control_memory[base_addr_of_FCB];
        }
        else if(*(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28] == global_biggest_size){
          if (*(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 20] < latest_time && *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 20] > global_latest_time){
            biggest_FCB = *(int*)&fs->directory_control_memory[base_addr_of_FCB];
            biggest_size = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28];
            latest_time = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 20];
          }
        }
      }
    }
    if (biggest_size < global_biggest_size){
      global_biggest_size = biggest_size;
      latest_time = 0X7FFF;
      global_latest_time = -1;
      i -= 1;
      continue;
    }
    int j = 0;
    do{
      printf("%c", fs->volume[biggest_FCB+j]);
      j += 1;
    }while (fs->volume[biggest_FCB+j] != '\0');
    if (fs->volume[biggest_FCB + fs->FCB_SIZE-1] == 3){    /* when it is a directory */
      printf(" %d d\n", biggest_size);
    }else{
      printf(" %d\n", biggest_size);
    }

    global_latest_time = latest_time;
    latest_time = 0x7FFF;

    global_biggest_size = biggest_size;
    biggest_size = 0;
  }
}

__device__ int check_total_all_files(FileSystem *fs){
  int total = 0;
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
      total += 1;
    }
  }
  return total;
}

/* This function is implemented to list all the files and directory whatever it is in which directory */
/* I use this function to check whether RM-RF deletes files and directories correctly */
__device__ void list_all_size(FileSystem *fs){
  u32 biggest_size = 0;
  u32 global_biggest_size = 0x7FFF;
  u32 biggest_FCB = 0;
  u32 latest_time = 0x7FFF;
  u32 global_latest_time = 0;
  int file_numbers = check_total_all_files(fs);
  /* if the sizes are the same, sort by the create time */
  for (int i = 0; i < file_numbers; i++){
    for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
      if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
        if (*(short*)&fs->volume[base_addr_of_FCB + 28] > biggest_size && *(short*)&fs->volume[base_addr_of_FCB + 28] < global_biggest_size){
          biggest_size = *(short*)&fs->volume[base_addr_of_FCB + 28];
          biggest_FCB = base_addr_of_FCB;
        }
        else if(*(short*)&fs->volume[base_addr_of_FCB + 28] == global_biggest_size){
          if (*(short*)&fs->volume[base_addr_of_FCB + 20] < latest_time && *(short*)&fs->volume[base_addr_of_FCB + 20] > global_latest_time){
            biggest_FCB = base_addr_of_FCB;
            biggest_size = *(short*)&fs->volume[base_addr_of_FCB + 28];
            latest_time = *(short*)&fs->volume[base_addr_of_FCB + 20];
          }
        }
      }
    }
    if (biggest_size < global_biggest_size){
      global_biggest_size = biggest_size;
      latest_time = 0X7FFF;
      global_latest_time = 0;
      i -= 1;
      continue;
    }
    int j = 0;
    do{
      printf("%c", fs->volume[biggest_FCB+j]);
      j += 1;
    }while (fs->volume[biggest_FCB+j] != '\0');
    printf(" %d\n", biggest_size);

    global_latest_time = latest_time;
    latest_time = 0x7FFF;

    global_biggest_size = biggest_size;
    biggest_size = 0;
  }
}

/* this function is mainly used for finding the free block in and return its block number */
__device__ int find_and_allocate_free_block(FileSystem *fs, int num_of_block){
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
    for (int bit_index = 128; bit_index > 0; bit_index /=2){
      if (!(bit_index & fs->volume[i])){   /* means the free block is founded */
          if (num_of_block == 32) printf("here\n");
          int current_bit_vector = i;
          int current_bit_index = bit_index;
          for (int bit = 0; bit < num_of_block; bit ++){
            fs->volume[current_bit_vector] = fs->volume[current_bit_vector] | current_bit_index;     /* allocate this block to the new file */
            if (current_bit_index == 1){
              current_bit_index = 128;
              current_bit_vector += 1;
            }else{
              current_bit_index /= 2;
            }
          }
          if (bit_index == 128) return (8*i + 0);
          else if (bit_index == 64) return (8*i + 1);
          else if (bit_index == 32) return (8*i + 2);
          else if (bit_index == 16) return (8*i + 3);
          else if (bit_index == 8) return (8*i + 4);
          else if (bit_index == 4) return (8*i + 5);
          else if (bit_index == 2) return (8*i + 6);
          else return (8*i + 7);
      }
    }
  }
  printf("The the data content is full!\n");
  return -1;
}


__device__ void compact(FileSystem *fs, short exit_size, u32 exit_pointer){

  /* first, calculate the exit block size */
  int block_number = (int)exit_size / fs->STORAGE_BLOCK_SIZE;
  if ((int)exit_size % fs->STORAGE_BLOCK_SIZE > 0 || exit_size == 0) block_number += 1;

  /* then compact the data by copy*/
  size_t num_move_byte = fs->STORAGE_SIZE - exit_pointer - fs->STORAGE_BLOCK_SIZE*block_number;
  for (int i = 0; i < num_move_byte; i++){
    fs->volume[exit_pointer+i] = fs->volume[exit_pointer+fs->STORAGE_BLOCK_SIZE*block_number + i];
  }
  /* update the FCBs */
  u32 last_block = exit_pointer;
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    /* check whether the pointer is behind the exit pointer */
    if (check_pointer(fs, exit_pointer, base_addr_of_FCB+24) && fs->volume[base_addr_of_FCB + fs->FCB_SIZE-1] == 1){  /* means its data has been moved forward */
      if (last_block < *(u32*)&fs->volume[base_addr_of_FCB+24]){
        last_block = *(u32*)&fs->volume[base_addr_of_FCB+24];
      }
      *(u32*)&fs->volume[base_addr_of_FCB+24] -= block_number*fs->FCB_SIZE;
    }
  }
  /* update the bitmap */
  int modified_char_num = block_number / 8;      /* the number of bit vector need to be modified */
  if (block_number % 8 > 0) modified_char_num += 1;

  int last_block_index = (last_block - fs->SUPERBLOCK_SIZE - fs->FCB_ENTRIES*fs->FCB_SIZE) / fs->STORAGE_BLOCK_SIZE;
  if ((last_block - fs->SUPERBLOCK_SIZE - fs->FCB_ENTRIES*fs->FCB_SIZE) % fs->STORAGE_BLOCK_SIZE != 0) printf("error: not aligned 2!\n");
  int bit_vector = last_block_index / 8;

  for (int j = modified_char_num-1; j >= 0; j--){
    if (block_number > 8){
      fs->volume[bit_vector-j] = fs->volume[bit_vector-j] << 8;
      block_number -= 8;
    }else{
      fs->volume[bit_vector-j] = fs->volume[bit_vector-j] << block_number;
    }
  }
}

__device__ int find_free__DCB(FileSystem *fs){
  for (int i = 0; i < 1024*DCB_SIZE; i += DCB_SIZE){
    if (*(uchar*)&fs->directory_control_memory[i + DCB_SIZE-1] == 0){
      *(uchar*)&fs->directory_control_memory[i + DCB_SIZE-1] = 1;
      *(int*)&fs->directory_control_memory[i + 4] = fs->current_dir;
      return i;
    }
  }
  printf("not found empty DCB\n");
  return -1;
}

__device__ int find_empty_DCB(FileSystem *fs){
  for (int i = fs->current_dir*DCB_SIZE + 8; i < fs->current_dir*DCB_SIZE+ DCB_SIZE; i += 4){
    if (*(int*)&fs->directory_control_memory[i] == 0){
      return i;
    }
  }
  printf("exceed 50 files in directory!\n");
  return -1;
}

/* open a file s in read/write mode according to the 
file name if for write mode, the file doesn't exist, 
create one. Finally return the file pointer      */
__device__ u32 fs_open(FileSystem *fs, char *s, int op, short create_time)
{
  /* First, it will search for the filename in the FCB */
  int find_filename_in_FCB = 0;
  u32 base_addr_of_FCB;
  for (base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE + DCB_SIZE - 4; base_addr_of_FCB+=4){
    if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] == 1){
      if (check_name(s, &fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]])){
        find_filename_in_FCB = 1;      /* the filename already exist in the FCB */
        break;
      }
    }
  }
  if (find_filename_in_FCB){   /* if the filename is found in FCB, return the pointer to the data content */
    return *(int*)&fs->directory_control_memory[base_addr_of_FCB];
  }else{                      /* if the filename is not founded */
    if (op == G_READ){
      printf("file not found!\n");
      return -1;
    }
    else if (op == G_WRITE || op == G_CREATE_D){   /* if the file/directory doesn't exist, create a new file/directory */
      /* it will find the bitmap in super block to find the free block */
      int free_block;
      if (op == G_WRITE) {        
        free_block = find_and_allocate_free_block(fs, 1);  /* by default, allocate it 1 block */
      }
      else if (op == G_CREATE_D){
        free_block = find_free__DCB(fs);
      }
      /* allocate this block to the new file */
      u32 free_FCB = find_free_FBC(fs);      /* this is the address of the FBC */

      /* store this pointer to the directory control memory */
      int free_one = find_empty_DCB(fs);
      *(int*)&fs->directory_control_memory[free_one] = free_FCB;

      int char_index = 0;
      do
      {
        fs->volume[free_FCB+char_index] = s[char_index];
        char_index += 1;
      }while(s[char_index] != '\0');
      char_index += 1;
      
      if(fs->current_dir > 0) *(short*)&fs->volume[*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE] + 28] += char_index;
      if (create_time > 0){
        *(short*)&fs->volume[free_FCB + 20] = create_time;
        *(short*)&fs->volume[free_FCB + 22] = gtime;       /* when create, the modified time is the same */
      }else{
        *(short*)&fs->volume[free_FCB + 20] = gtime;       /* store the create time */
        *(short*)&fs->volume[free_FCB + 22] = gtime;       /* when create, the modified time is the same */
      }
      gtime += 1;
      *(short*)&fs->volume[free_FCB + 28] = 0;     /* set file size = 0B */
      u32 file_pointer; 
      if (op == G_WRITE){
        file_pointer = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fs->FCB_ENTRIES + fs->STORAGE_BLOCK_SIZE*free_block;
        *(u32*)&fs->volume[free_FCB + 24] = file_pointer;
        fs->volume[free_FCB+fs->FCB_SIZE-1] = 1;
        /* when create a new file, put its FCB pointer into the directory control memory */
      }
      else if (op == G_CREATE_D){
        file_pointer = free_block;
        *(u32*)&fs->volume[free_FCB + 24] = file_pointer;
        fs->volume[free_FCB+fs->FCB_SIZE-1] = 3;
        *(int*)&fs->directory_control_memory[free_block] = free_FCB;
      }
      return free_FCB;
    }
  }
}


/* To read the buffer from the file which is pointed by u32 fp
And store the data into the ucahr * output */
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* first read all the data out */
  u32 datapointer = *(u32*)&fs->volume[fp + 24];
	for (u32 i = 0; i < size; i++){
    output[i] = fs->volume[datapointer + i];
  }
  /* then set the access time to the newest time */

}


/* u32 fp is a pointer points to the file,
if file exist, cleanup the older contents and wrtie the new content 
the new content is in the input buffer uchar * input */
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  /* first it will look into FCB and find the allocated size*/
  short allocated_size;
  char* filename = (char*)malloc(fs->MAX_FILENAME_SIZE);
  int needed_block = size / fs->STORAGE_BLOCK_SIZE;     /* represents how many blocks this size occupy */
  if (size % fs->STORAGE_BLOCK_SIZE > 0) needed_block += 1;
  u32 datapointer = *(u32*)&fs->volume[fp + 24];
  u32 base_addr_of_FCB;
  for (base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE+DCB_SIZE-4; base_addr_of_FCB += 4){
    if (fp == *(int*)&fs->directory_control_memory[base_addr_of_FCB]){
      *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 22] = gtime;     /* store the modified time */
      gtime += 1;
      allocated_size = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28];
      int number_block = allocated_size / fs->STORAGE_BLOCK_SIZE;
      if (allocated_size % fs->STORAGE_BLOCK_SIZE > 0 || allocated_size == 0) number_block += 1;
      allocated_size = number_block * fs->STORAGE_BLOCK_SIZE;
      int i = 0;
      do{
        filename[i] = fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]+i];
        i += 1;
      }while (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]+i] != '\0');
      filename[i] = '\0';
      break;
    }
  }
  if (allocated_size == needed_block*fs->STORAGE_BLOCK_SIZE) {         /* the allocated size match the size */
    for (u32 i = 0; i < size; i++){
      fs->volume[datapointer + i] = input[i];
    }
    *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28] = size;     /* update the size of it */
  }else{                                /* the size is not enough, then, need to find a larger place */
    short create_time = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 20];
    /* we delete the original place first */
    fs_gsys(fs, RM, filename);
    /* then we assign a enough place for it */
    u32 new_fp = *(u32*)&fs->volume[fs_open(fs, filename, G_WRITE, create_time) + 24];  /*since the storage has been compacted, it will be assigned with the last unused block */
    /* change the size attibute in FBCs */
    for (base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE+DCB_SIZE -4; base_addr_of_FCB += 4){
      if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] == 1){
        if (check_name(filename, &fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]])){
          *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28] = size;
          break;
        }
      }
    }
    find_and_allocate_free_block(fs, needed_block-1);  /* allocate the remaining blocks */
    /* write data into the storage */
    for (u32 i = 0; i < size; i++){
      fs->volume[new_fp + i] = input[i];
    }
  }
}


/* LS_D: list all the file name in the directory and order by modified time of files */
/* LS_S: list all the file name and size in the directory order by size */
__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  if (op == LS_D){           /* list the files by time */
    printf("===sort by modified time=== \n");
    list_by_time(fs);
  }
  else if(op == LS_S){
    printf("===sort by file size=== \n");
    list_by_size(fs);
  }
  else if (op == CD_P){
    fs->current_dir = *(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE+4];
  }
  else if (op == PWD){
    int father_directory = *(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 4];
    if (father_directory != 0){
      printf("/");
      int i = 0;
      int FCB_addr = *(int*)&fs->directory_control_memory[father_directory*DCB_SIZE];
      do{
        printf("%c", fs->volume[FCB_addr + i]);
        i += 1;
      }while (i < 20 && fs->volume[FCB_addr + i] != '\0');
    }
    printf("/");
    int j = 0;
    int FCB_addr = *(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE];
    do{
      printf("%c", fs->volume[FCB_addr + j]);
      j += 1;
    }while (j < 20 && fs->volume[FCB_addr + j] != '\0');
    printf("\n");
  }
  else if (op == LIST_ALL){
    list_all_size(fs);
  }
}

/* this function is used to delete a directory only have files */
__device__ void delete_dir(FileSystem * fs, int target_dir){
  int name_size = 0;
  int father_dcb = *(int*)&fs->directory_control_memory[target_dir + 4];
  int father_fcb = *(int*)&fs->directory_control_memory[father_dcb*DCB_SIZE];
  int fcb = *(int*)&fs->directory_control_memory[target_dir];

  /* first, we delete the files in dir one by one */
  for (int index = 0; index < 49; index++){

    if (*(int*)&fs->directory_control_memory[target_dir + 8 + index*4] != 0 && fs->volume[*(int*)&fs->directory_control_memory[target_dir + 8 + index*4] + fs->FCB_SIZE-1] == 1){  /* recognize it is a file */
      /* the pointer points to DCB */
      u32 base_addr_of_FCB = target_dir + 8 + index*4;
      u32 file_pointer = *(u32*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 24];
      short file_size = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28];
      for (int i = 0; i < fs->FCB_SIZE-1; i++){
        fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + i] = NULL;
      }
      fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE-1] = 0;    /* set the valid bit to 0 */
      /* then, clear the data */
      for (int i = 0; i < file_size; i ++){
        fs->volume[file_pointer+i] = NULL;
      }
      /* then delete it in the directory structure */
      *(int*)&fs->directory_control_memory[base_addr_of_FCB] = 0;
      /* then, do the compaction, during the compacting, bitmap will be updated */
      compact(fs, file_size, file_pointer);
    }
  }
  /* before we clear the dir, we reduce the size of its father */
  do{
    name_size += 1;
  }while(fs->volume[fcb+name_size] != '\0');
  name_size += 1;
  *(short*)&fs->volume[father_fcb + 28] -= name_size;

  /* then, we delelte the dir in the FCB */
  for (int index = 0; index < 32; index++){
    fs->volume[*(int*)&fs->directory_control_memory[target_dir] + index] = 0;
  }
  
  /* finally, clear the all the bit in directory block */
  for (int index = 0; index < DCB_SIZE; index++){
    fs->directory_control_memory[target_dir + index] = 0;
  }
}

/* remove the file according to the file name */
/* also, need to delete FCB, after that, need to do compaction */
__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  if (op == RM){
    /* First, Find the corresponding file in FCB */
    int find_filename_in_FCB = 0;
    int base_addr_of_FCB;
    for (base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE+DCB_SIZE - 4; base_addr_of_FCB += 4){
      if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] == 1){
        if (check_name(s, &fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]])){
          find_filename_in_FCB = 1;      /* the filename already exist in the FCB */
          break;
        }
      }
    }
    /* if the file is not found, print a message */    
    if (!find_filename_in_FCB) {
      printf("there is no such file: ");
      int i = 0;
      do{
        printf("%c",s[i]);
        i += 1;
      }while(s[i] != '\0' && i < 20);
      printf("\n");
    }
    else{  /* if the file is founded, stores its pointer, then clear message in FBC and set the valid bit to 0 */
      u32 file_pointer = *(u32*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 24];
      short file_size = *(short*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + 28];
      for (int i = 0; i < fs->FCB_SIZE-1; i++){
        fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + i] = NULL;
      }
      fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE-1] = 0;    /* set the valid bit to 0 */
      /* then, clear the data */
      for (int i = 0; i < file_size; i ++){
        fs->volume[file_pointer+i] = NULL;
      }
      /* then delete it in the directory structure */
      *(int*)&fs->directory_control_memory[base_addr_of_FCB] = 0;
      int i = 0;
      do{
        i += 1;
      }while(s[i] != '\0' && i < 20);
      i += 1;
      *(short*)&fs->volume[*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE] + 28] -= i;
      /* then, do the compaction, during the compacting, bitmap will be updated */
      compact(fs, file_size, file_pointer);
    }
  }
  else if (op == MKDIR){
    u32 fp = fs_open(fs, s, G_CREATE_D);    /* initialize a new directory here */
  }
  else if (op == CD){
    int base_addr_of_FCB;
    for (base_addr_of_FCB = fs->current_dir*DCB_SIZE+8; base_addr_of_FCB < fs->current_dir*DCB_SIZE+DCB_SIZE-4; base_addr_of_FCB += 4){
      if (fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB] + fs->FCB_SIZE -1] == 3){ /* if it is 3, it is a dir */
        if (check_name(s, &fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]])){
          fs->current_dir = *(int*)&fs->volume[*(int*)&fs->directory_control_memory[base_addr_of_FCB]+24] / DCB_SIZE;    /* change the current dir to the new one */
          return ;
        }
      }
    }
    printf("no such directory!\n");
  }
  else if (op == RM_RF){
    /* before all, we find the directory */
    int target_dir = -1;       /* base address of the DCB */
    int third_dir = -1;
    int index;
    for (index = 0; index < 49; index++){
      if (*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 8 + index*4] != 0 && fs->volume[*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 8 + index*4] + fs->FCB_SIZE -1] == 3){
        if (check_name(s, &fs->volume[*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 8 + index*4]])){
          target_dir = *(int*)&fs->volume[*(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 8 + index*4] + 24];
          break;
        }
      }
    }
    if (target_dir == -1) printf("no such diretory!\n");

    /* first, we find whether there is a sub-directory under it */
    /* if there is, goes into the third diectory */
    for (int j = 0; j < 49; j++){
      if (*(int*)&fs->directory_control_memory[target_dir+8+j*4] != 0 && fs->volume[*(int*)&fs->directory_control_memory[target_dir+8+j*4] + fs->FCB_SIZE-1] == 3){
        third_dir = *(int*)&fs->volume[*(int*)&fs->directory_control_memory[target_dir+8+j*4] + 24];  
        /* we assume that, third directory doesn't have sub-directory any more */
        delete_dir(fs, third_dir);
        /* then, delelet it in the second directory  and recude the size */
        *(int*)&fs->directory_control_memory[third_dir+8+j*4] = 0;
      }
    }
    /* after this for, all the sub-directory has been deleted, now delelte the files in target directory */
    delete_dir(fs, target_dir);
    /* then delete it in the first directory*/
    *(int*)&fs->directory_control_memory[fs->current_dir*DCB_SIZE + 8 + index*4] = 0;
  }
}
