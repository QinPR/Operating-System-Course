#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){
	int status;
	/* fork a child process */
	printf("Process start to fork\n");
	pid_t pid = fork();
	if (pid < 0){
		printf("Occured fork error \n");
	}
	/* execute test program */ 
	else{
		/* Child process */
		if (pid == 0){
			printf("I'm the Child Process, my pid = %d\n", getpid());
			char *arg[argc];
			int i;
			for(i=0; i<argc-1; i++){
				arg[i] = argv[i+1];
			}
			arg[argc-1] = NULL;
			printf("Child process start to execute test program: \n");
			/* Execute the input testing program and send signals to parent process */
			execve(arg[0], arg, NULL);
			exit(EXIT_FAILURE);
		}
		/* Parent process: wait for child process terminates */
		else{
			printf("I'm the Parent Process, my pid = %d\n", getpid());
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");

			/* check child process' termination status and print the outcome to console */
			/* end normally */
			if(WIFEXITED(status)){   
            	printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
			/* termination signals */
            else if(WIFSIGNALED(status)){	
                if (WTERMSIG(status) == 14){
					printf("child process get SIGALRM signal\n");
					printf("child process is exited by sig_alarm\n");
				}
				else if (WTERMSIG(status) == 6){
					printf("child process get SIGABRT signal\n");
					printf("child process is aborted\n");
				}
				else if (WTERMSIG(status) == 7){
					printf("child process get SIGBUS signal\n");
					printf("child process is failed because of bus error\n");
				}
				else if (WTERMSIG(status) == 8){
					printf("child process get SIGFPE signal\n");
					printf("child process is failed because of floating point exception\n");
				}
				else if (WTERMSIG(status) == 1){
					printf("child process get SIGHUP signal\n");
					printf("child process is hung up\n");
				}
				else if (WTERMSIG(status) == 4){
					printf("child process get SIGILL signal\n");
					printf("child process is failed because of illegal instruction\n");
				}
				else if (WTERMSIG(status) == 2){
					printf("child process get SIGINT signal\n");
					printf("child process is interrupted\n");
				}
				else if (WTERMSIG(status) == 9){
					printf("child process get SIGKILL signal\n");
					printf("child process is killed\n");
				}
				else if (WTERMSIG(status) == 13){
					printf("child process get SIGPIPE signal\n");
					printf("child process is failed because of pipe error\n");
				}
				else if (WTERMSIG(status) == 3){
					printf("child process get SIGQUIT signal\n");
					printf("child process is quited\n");
				}
				else if (WTERMSIG(status) == 11){
					printf("child process get SIGSEGV signal\n");
					printf("child process is failed because of segment fault\n");
				}
				else if (WTERMSIG(status) == 15){
					printf("child process get SIGTERM signal\n");
					printf("child process is terminated\n");
				}
				else if (WTERMSIG(status) == 5){
					printf("child process get SIGTRAP signal\n");
					printf("child process is trapped\n");
				}
				else{
					printf("status: %d\n", WTERMSIG(status));
				}
				printf("CHILD EXECUTION FAILED\n");
				
            }
			/* being stopped */
            else if(WIFSTOPPED(status)){   
				printf("child process get SIGSTOP signal\n");
				printf("child process stopped\n");
                printf("CHILD PROCESS STOPPED\n");
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
		}
	}
}
