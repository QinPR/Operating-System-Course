#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

/* char arg[]: files to be execute by each child process 
   int fork_num: rank of child process (e.g. for the first child, fork_num = 0)
   int length: number of total child process
   int * p: pointers points to address of shared memory   */
void create_fork_link(char * arg[], int fork_num, int length, int * p){
	int status;

	if ((fork_num < length) && (arg[fork_num])){
		pid_t pid = fork();
		if (pid < 0){
			printf("Occured fork error \n");
		}
		else{
			if (pid == 0){			/* child process execute */
				fork_num += 1;
				*(p+8*fork_num) = (int)getpid();	/* store its pid to shared memory */
				create_fork_link(arg, fork_num, length, p);		/* create the child process's child process (recursion entry)*/
				execve(arg[fork_num-1], arg, NULL);
				exit(EXIT_FAILURE);
			}
			else{					/* parent process execute */
				waitpid(pid, &status, WUNTRACED);
				*(p+8*fork_num+4) = (int)status;	/* store its child return signal to shared memory */
			}
		}
	}
	return;
}

int main(int argc,char *argv[]){
	int *p;
	int status;
	char *arg[argc];
	int i;
	int fork_num;

	/* for the communication between two process, using a shared-memory strategy */
	p = (int*)mmap(NULL, 20*argc, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);	
	if (p == MAP_FAILED){
		printf("mmap error\n");
		exit(1);
	}
	*p = (int)getpid();
	for(i=0; i<argc-1; i++){
		arg[i] = argv[i+1];
	}
	arg[argc-1] = NULL;
	fork_num = 0;
	create_fork_link(arg, fork_num, argc, p);		/* fork a child process here */

	/* after all the child process terminates, print out the corresponding message */
	printf("\n---\n");
	printf("Process tree: ");
	for (int k = 0; k < argc; k++){
		if (k == argc-1){
			printf("%d\n", *(p+k*8));
		}
		else{
			printf("%d->", *(p+k*8));
		}
	}
	for (int k = argc-1; k >= 0; k--){
		if (k != 0){
			status = *(p+k*8-4);
			printf("Child process %d of parent process %d ",(int)*(p+k*8), (int)*(p+k*8-8));
			/* child process ends normally */
			if(WIFEXITED(status)){   
				printf("terminated normally with exit code %d\n", WEXITSTATUS(status));
            }
			/* child process terminates */
            else if(WIFSIGNALED(status)){	
                if (WTERMSIG(status) == 14){
					printf("is terminated by signal %d (Alarm clock)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 6){
					printf("is terminated by signal %d (Abort)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 7){
					printf("is terminated by signal %d (Bus error)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 8){
					printf("is terminated by signal %d (Floating point exception)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 1){
					printf("is terminated by signal %d (Hang up)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 4){
					printf("is terminated by signal %d (Illegal instruction)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 2){
					printf("is terminated by signal %d (Interrupt)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 9){
					printf("is terminated by signal %d (Kill)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 13){
					printf("is terminated by signal %d (Pipe)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 3){
					printf("is terminated by signal %d (Quit)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 11){
					printf("is terminated by signal %d (Segment fault)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 15){
					printf("is terminated by signal %d (Terminate)\n", WTERMSIG(status));
				}
				else if (WTERMSIG(status) == 5){
					printf("is terminated by signal %d (Trap)\n", WTERMSIG(status));
				}
				else{
					printf("status: %d\n", WTERMSIG(status));
				}
				
            }
			/* child process stops or continues to execute */
            else if(WIFSTOPPED(status)){  
				printf("is stopped by signal %d (Stop)\n", WSTOPSIG(status));
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
		}
		else{
			printf("Myfork process (%d) terminated normally\n", *(p));
		}
		
	}
	int ret = munmap(p, 200);		/* release the shared memory */
	if (ret == -1){
		printf("munmap error\n");
		exit(1);
	}

	return 0;
}
