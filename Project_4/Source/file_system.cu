#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

__device__ __managed__ u32 gtime = 0;

__device__ void init_FBC(FileSystem *fs){
  for (int base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    fs->volume[base_addr_of_FCB+fs->FCB_SIZE-1] == 0;
  }
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
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
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
      total += 1;
    }
  }
  return total;
}

__device__ void list_by_time(FileSystem *fs){
  u32 latest_time = 0;
  u32 global_latest_time = 0xFFFFFFFF;
  u32 latest_FCB = 0;
  int file_numbers = check_total_files(fs);
  for (int i = 0; i < file_numbers; i++){
    for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
      if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
        if (*(short*)&fs->volume[base_addr_of_FCB + 22] > latest_time && *(short*)&fs->volume[base_addr_of_FCB + 22] < global_latest_time){
          latest_time = *(short*)&fs->volume[base_addr_of_FCB + 22];
          latest_FCB = base_addr_of_FCB;
        }
      }
    }
    global_latest_time = latest_time;
    latest_time = 0;
    int j = 0;
    do{
      printf("%c", fs->volume[latest_FCB+j]);
      j += 1;
    }while (fs->volume[latest_FCB+j] != '\0');
    printf("\n");
  }
}

__device__ void list_by_size(FileSystem *fs){
  u32 biggest_size = 0;
  u32 global_biggest_size = 0x7FFF;
  u32 biggest_FCB = 0;
  u32 latest_time = 0x7FFF;
  u32 global_latest_time = 0;
  int file_numbers = check_total_files(fs);
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

  /* know how many bytes should be moved forward */
  u32 biggest_pt = fs->FILE_BASE_ADDRESS;
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    /* check whether the pointer is behind the exit pointer */
    if (*(u32*)&fs->volume[base_addr_of_FCB + 24] > biggest_pt){  /* means its data has been moved forward */
      biggest_pt = *(u32*)&fs->volume[base_addr_of_FCB + 24];
    }
  }
  biggest_pt += 2048;

  /* then compact the data by copy*/
  size_t num_move_word = (biggest_pt - exit_pointer - fs->STORAGE_BLOCK_SIZE*block_number)/4;
  if ((biggest_pt - exit_pointer - fs->STORAGE_BLOCK_SIZE*block_number) % 4 != 0) printf("warning!!!!!!!!!!");
  for (int i = 0; i < num_move_word; i++){
    *(u32*)&fs->volume[exit_pointer+i*4] = *(u32*)&fs->volume[exit_pointer+fs->STORAGE_BLOCK_SIZE*block_number + i*4];
  }
  /* update the FCBs */
  u32 last_block = exit_pointer;
  for (u32 base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    /* check whether the pointer is behind the exit pointer */
    if (check_pointer(fs, exit_pointer, base_addr_of_FCB+24)){  /* means its data has been moved forward */
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

/* open a file s in read/write mode according to the 
file name if for write mode, the file doesn't exist, 
create one. Finally return the file pointer      */
__device__ u32 fs_open(FileSystem *fs, char *s, int op, short create_time)
{
  /* First, it will search for the filename in the FCB */
  int find_filename_in_FCB = 0;
  u32 base_addr_of_FCB;
  for (base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
      if (check_name(s, &fs->volume[base_addr_of_FCB])){
        find_filename_in_FCB = 1;      /* the filename already exist in the FCB */
        break;
      }
    }
  }
  if (find_filename_in_FCB){   /* if the filename is found in FCB, return the pointer to the data content */
    return base_addr_of_FCB;
    // return *(u32*)&fs->volume[(base_addr_of_FCB + 24)];  /* the pointer to the data content has offset = 28 in FCB */
  }else{                      /* if the filename is not founded */
    if (op != G_WRITE){
      printf("file not found!\n");
      return -1;
    }else{                    /* if the file doesn't exist, create a new file */
    
      /* it will find the bitmap in super block to find the free block */
      int free_block = find_and_allocate_free_block(fs, 1);  /* by default, allocate it 1 block */
      /* allocate this block to the new file */
      u32 free_FCB = find_free_FBC(fs);
      int char_index = 0;
      do
      {
        fs->volume[free_FCB+char_index] = s[char_index];
        char_index += 1;
      }while(s[char_index] != '\0');
      if (create_time > 0){
        *(short*)&fs->volume[free_FCB + 20] = create_time;
        *(short*)&fs->volume[free_FCB + 22] = gtime;       /* when create, the modified time is the same */
      }else{
        *(short*)&fs->volume[free_FCB + 20] = gtime;       /* store the create time */
        *(short*)&fs->volume[free_FCB + 22] = gtime;       /* when create, the modified time is the same */
      }
      gtime += 1;
      *(short*)&fs->volume[free_FCB + 28] = 0;     /* set file size = 0B */ 
      u32 file_pointer = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fs->FCB_ENTRIES + fs->STORAGE_BLOCK_SIZE*free_block;
      *(u32*)&fs->volume[free_FCB + 24] = file_pointer;
      fs->volume[free_FCB+fs->FCB_SIZE-1] = 1;

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
  short file_size = *(short*)&fs->volume[fp + 28];
  if (file_size < size){
    printf("error: read %d bytes from file with %d bytes\n", size, file_size);
    return;
  }
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
  for (base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
    if (datapointer == *(u32*)&fs->volume[base_addr_of_FCB + 24]){
      *(short*)&fs->volume[base_addr_of_FCB + 22] = gtime;     /* store the modified time */
      gtime += 1;
      allocated_size = *(short*)&fs->volume[base_addr_of_FCB + 28];
      int number_block = allocated_size / fs->STORAGE_BLOCK_SIZE;
      if (allocated_size % fs->STORAGE_BLOCK_SIZE > 0 || allocated_size == 0) number_block += 1;
      allocated_size = number_block * fs->STORAGE_BLOCK_SIZE;
      int i = 0;
      do{
        filename[i] = fs->volume[base_addr_of_FCB+i];
        i += 1;
      }while (fs->volume[base_addr_of_FCB+i] != '\0');
      break;
    }
  }
  if (allocated_size == needed_block*fs->STORAGE_BLOCK_SIZE) {         /* the allocated size match the size */
    for (u32 i = 0; i < size; i++){
      fs->volume[datapointer + i] = input[i];
    }
    *(short*)&fs->volume[base_addr_of_FCB + 28] = size;     /* update the size of it */
  }else{                                /* the size is not enough, then, need to find a larger place */
    short create_time = *(short*)&fs->volume[base_addr_of_FCB + 20];
    /* we delete the original place first */
    fs_gsys(fs, RM, filename);
    /* then we assign a enough place for it */

    u32 new_fp = *(u32*)&fs->volume[fs_open(fs, filename, G_WRITE, create_time) + 24];  /*since the storage has been compacted, it will be assigned with the last unused block */
    /* change the size attibute in FBCs */
    for (base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
      if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
        if (check_name(filename, &fs->volume[base_addr_of_FCB])){
          *(short*)&fs->volume[base_addr_of_FCB + 28] = size;
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
}


/* remove the file according to the file name */
/* also, need to delete FCB, after that, need to do compaction */
__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  if (op == RM){
    /* First, Find the corresponding file in FCB */
    int find_filename_in_FCB = 0;
    int base_addr_of_FCB;
    for (base_addr_of_FCB = fs->SUPERBLOCK_SIZE; base_addr_of_FCB < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE; base_addr_of_FCB+=fs->FCB_SIZE){
      if (fs->volume[base_addr_of_FCB + fs->FCB_SIZE -1] == 1){
        if (check_name(s, &fs->volume[base_addr_of_FCB])){
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
      u32 file_pointer = *(u32*)&fs->volume[base_addr_of_FCB + 24];
      short file_size = *(short*)&fs->volume[base_addr_of_FCB + 28];
      for (int i = 0; i < fs->FCB_SIZE-1; i++){
        fs->volume[base_addr_of_FCB + i] = NULL;
      }
      fs->volume[base_addr_of_FCB + fs->FCB_SIZE-1] = 0;    /* set the valid bit to 0 */
      /* then, clear the data */
      for (int i = 0; i < file_size; i ++){
        fs->volume[file_pointer+i] = NULL;
      }
      /* then, do the compaction, during the compacting, bitmap will be updated */
      compact(fs, file_size, file_pointer);
    }
  }
}
