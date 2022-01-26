#include <QApplication>
#include <QLabel>
#include <QWidget>
#include <QPalette>
#include <QTimer>
#include <QLabel>
#include <QPaintEvent>
#include <QPainter>
#include <QKeyEvent>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include "hw2.h"

#define ROW 10
#define COLUMN 50 
#define NUM_THREAD 10
#define NUM_LOGS 9

int thread_ids[NUM_THREAD] = {0,1,2,3,4,5,6,7,8,9};
pthread_mutex_t mutex_frog;		/* mutex lock to protect frog */
pthread_mutex_t mutex_map;		/* mutex lock to protect map */
pthread_mutex_t mutex_dir;		/* mutex lock to protect the control of frog */
int status = 1;   				/* indicate why the game is end. status = 2: win; = 1: continue; = 0: lose */
int speed = 70000;
int log_length[NUM_LOGS];
int start_point[NUM_LOGS];
int number_of_moved_log = 0;	/* 0 means all the logs haven't start moving, 9 means all the logs have moved in one iteration */
char frog_dir;
int game_end = 0;				/* indicate the end of game */

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
}; 

Node frog = Node( ROW, (COLUMN-1) / 2 ); 
char map[ROW+10][COLUMN] ;

/* each thread of logs will invoke this function individually*/
/* this thread performs the move of each log */
void *logs_move( void *rank ){
	int *my_id = (int*)rank;
	start_point[*my_id] = rand() % (COLUMN-1);		/* random initialize the appear position of log */
	int direction = *my_id % 2;
	log_length[*my_id] = rand() % 20;				/* random initialize the length, limit to < 20 and > 10 */

	while(log_length[*my_id] < 10){
		log_length[*my_id] = rand() % 20;
	}
	pthread_mutex_lock(&mutex_map);					/* put the positon of log onto map */
	for (int pos_x = 0; pos_x < log_length[*my_id]; pos_x++){
		map[*my_id+1][(start_point[*my_id] + pos_x)%(COLUMN-1)] = '=';
	}
	pthread_mutex_unlock(&mutex_map);	
	while(!game_end){
		usleep(speed);								/* this sleep time determine the speed of log movement */
		pthread_mutex_lock(&mutex_map);	
		if (direction){     /* move right */
			map[*my_id+1][start_point[*my_id]%(COLUMN-1)] = ' '; 
			map[*my_id+1][(start_point[*my_id]+log_length[*my_id])%(COLUMN-1)] = '=';
			start_point[*my_id] = (start_point[*my_id] + 1)%(COLUMN-1);
			pthread_mutex_lock(&mutex_frog);
			if (frog.x == *my_id+1){				/* if frog is on this log, move frog together */
				frog.y += 1;
			}
			pthread_mutex_unlock(&mutex_frog);
		}
		else{				/* move left */
			start_point[*my_id] = (49 + start_point[*my_id]-1)%(COLUMN-1);
			map[*my_id+1][start_point[*my_id]] = '='; 
			map[*my_id+1][(start_point[*my_id]+log_length[*my_id])%(COLUMN-1)] = ' ';
			pthread_mutex_lock(&mutex_frog);
			if (frog.x == *my_id+1){
				frog.y -= 1;
			}
			number_of_moved_log += 1;
			pthread_mutex_unlock(&mutex_frog);
		}
		pthread_mutex_unlock(&mutex_map);
	}
}

/* the main thread would be in charge of the refreshing of screen */
/* this refreshing process must be atomic, so two locks are used */
void MainWidget::printmap(){
	if (status == 1){
		char current_row[COLUMN] ;
		pthread_mutex_lock(&mutex_map);			/* lock both map and frog */
			if (number_of_moved_log == NUM_LOGS || number_of_moved_log == 0){/* use this to make sure that during each */
				number_of_moved_log = 0;									 /* time of refreshing, the logs can move simultanuously*/
			}else{
				pthread_mutex_unlock(&mutex_map);
			}
		pthread_mutex_lock(&mutex_frog);
		/* judge whether the game come to end */
		if (map[frog.x][frog.y] == ' ' || frog.y < 1 || frog.y > 47){		/* drop to river or reach the boundaries */
			status = 0;
			pthread_mutex_unlock(&mutex_frog);
			pthread_mutex_unlock(&mutex_map);
		}
		if (frog.x == 0){													/* win */
			status = 2;
			pthread_mutex_unlock(&mutex_frog);
			pthread_mutex_unlock(&mutex_map);
		}
		/* re-create the row that the frog is in */
		for (int i = 0; i < (COLUMN-1); i++){
			current_row[i] = map[frog.x][i];
		}
		current_row[frog.y] = '0';
		for( int i = 0; i <= ROW; ++i){										/* print the map onto the screen */
			int count = 0;
			if (i > 0 && i < ROW){
				for (int j = 0; j < COLUMN-1; j++){
					if (map[i][j] == '='){
						log[i-1][count]->move(j*10 + 20 , log[i-1][count]->y());
						count += 1;
					}
				}
			}
		}	
		frog_gui->move(frog.y*10 + 20, frog.x*10 + 5);						/* print the frog onto the screen */
		pthread_mutex_unlock(&mutex_frog);
		pthread_mutex_unlock(&mutex_map);
	}
	else if (status == 2){									/* when the game come to end, print the message on the screen */
		game_end = 1;
		game_msg->setText("you win the game!");
	}
	else if (status == 0){
		game_end = 1;
		game_msg->setText("you lose the game!");
	}
	else if (status == 3){
		game_end = 1;
		game_msg->setText("you quit!");
	}
}


void MainWidget::setspeedValue(int value){					/* change the speed of movement by slide bar */
  speed = (120-slider->value())*1000;
}

/* This part will be in charge of the graphic output */
MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent){
	/* initialize the slide bar */
	slider = new QSlider(this);
	slider->setOrientation(Qt::Horizontal);
	slider->resize(200, 200);
	slider->setMinimum(40);
	slider->setMaximum(110);
  	slider->setValue(70);
	slider->move(180, 40);
	/* put the slide bar and txt*/
	connect(slider, SIGNAL(valueChanged(int)), this, SLOT(setspeedValue(int)));
	speed_text = new QLabel(this);
	speed_text->setGeometry(123, 123, 50, 30);
	speed_text->setText("SPEED: ");
	/* initialize of the game msg */
	game_msg = new QLabel(this);
	game_msg->setGeometry(200, 40, 300, 50);

	/*initialize of the bounds*/
	upperbound = new QLabel(this);
	upperbound->setGeometry(20, frog.x*10+5, 490, 10); 
    upperbound->setStyleSheet("background-color:rbga(255,255,224,0%); border: 1px solid black");

	lowerbound = new QLabel(this);
	lowerbound->setGeometry(20, 5, 490, 10); 
    lowerbound->setStyleSheet("background-color:rbga(255,255,224,0%); border: 1px solid black");
	/*initialize of frog*/
	setFocus();
	frog_gui = new QLabel(this);
	frog_gui->setGeometry(frog.x*10+10, frog.y*10+10, 10, 10); 
    frog_gui->setStyleSheet("background-color:red; border-radius:25px");
	/* initialize of log*/
	usleep(100);
	for (int i = 0; i < NUM_LOGS; i++){
		for (int j = 0; j < log_length[i]; j++){
			log[i][j] = new QLabel(this);
			log[i][j]->setStyleSheet("border-width: 1px; border-style:solid; border-color:rgb(255, 170, 0); background-color:gray; border-radius:5px");
			log[i][j]->setGeometry(start_point[i]*10 + j*10 + 20, 10*i + 20, 10, 5);         // (x, y, length, width)
		}
	}
	QPalette palette;
    palette.setBrush(this->backgroundRole(), Qt::lightGray);
    this->setPalette(palette);
	connect(&timer, SIGNAL(timeout()), this, SLOT(printmap()));
    timer.start(20);     /* control the frequency of refreshing the screen */
}

/* listen to the keyborad */
void MainWidget::keyPressEvent(QKeyEvent * event){
    switch (event->key())
    {
        case Qt::Key_W:
			pthread_mutex_lock(&mutex_dir);
            frog_dir = 'W';
			pthread_mutex_unlock(&mutex_dir);
        	break;
        case Qt::Key_D:
            pthread_mutex_lock(&mutex_dir);
            frog_dir = 'D';
			pthread_mutex_unlock(&mutex_dir);
        	break;
        case Qt::Key_A:
            pthread_mutex_lock(&mutex_dir);
            frog_dir = 'A';
			pthread_mutex_unlock(&mutex_dir);
			break;
		case Qt::Key_S:
            pthread_mutex_lock(&mutex_dir);
            frog_dir = 'S';
			pthread_mutex_unlock(&mutex_dir);
        	break;
		case Qt::Key_Q:
            status = 3;
        	break;
    }
}

/* main thread will invoke this function and do the freshing of screen and graphic output */
int refresh_map(int argc, char** argv){
	QApplication app(argc, argv);
	MainWidget window;
	window.resize(600, 200);
  	window.setWindowTitle("Frog Game");
	window.show();
	return app.exec();
}

/* thread p_frog will call this function and control the frog */
void *f_frog(void*t){
	while (status == 1){				/* loop with no latency: make sure the control of frog can be responded as soon as possible */
		pthread_mutex_lock(&mutex_dir);
		if (frog_dir != '0'){
			if (frog_dir == 'W' || frog_dir == 'w'){
				pthread_mutex_lock(&mutex_frog);
				frog.x -= 1;
				pthread_mutex_unlock(&mutex_frog);
			}
			if (frog_dir == 'D' || frog_dir == 'd'){
				pthread_mutex_lock(&mutex_frog);
				frog.y = (frog.y + 1) % (COLUMN-1);
				pthread_mutex_unlock(&mutex_frog);
			}
			if (frog_dir == 'A' || frog_dir == 'a'){
				pthread_mutex_lock(&mutex_frog);
				frog.y = (49 + frog.y-1)%(COLUMN-1);
				pthread_mutex_unlock(&mutex_frog);
			}
			if (frog_dir == 'S' || frog_dir == 's'){
				pthread_mutex_lock(&mutex_frog);
				if (frog.x == ROW-1){			/* make sure when frog is on the botton, it won't go down anymore */
					frog.x += 1;
				}
				pthread_mutex_unlock(&mutex_frog);
			}
			if (frog_dir == 'Q' || frog_dir == 'q'){
				pthread_mutex_lock(&mutex_frog);
				status = 3;
				pthread_mutex_unlock(&mutex_frog);
			}
			if (frog_dir == ','){
				if (speed < 100000){
					speed += 3000;
				}
			}
			if (frog_dir == '.'){
				if (speed >= 50000){
					speed -= 5000;
				}
			}
			frog_dir = '0';
		}
		pthread_mutex_unlock(&mutex_dir);
	}
}

int main( int argc, char *argv[] ){
	pthread_t threads[NUM_THREAD];
	long rank;
	pthread_t refresh, p_frog;
	pthread_mutex_init(&mutex_frog, NULL);
	pthread_mutex_init(&mutex_map, NULL);
	pthread_mutex_init(&mutex_dir, NULL);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	// frog = Node( ROW, (COLUMN-1) / 2 );
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j ){
			map[i][j] = ' ' ;  
		}	
	}	
	for( j = 0; j < COLUMN - 1; ++j ){
		map[ROW][j] = map[0][j] = '|' ;
	}		
	for( j = 0; j < COLUMN - 1; ++j ){
		map[0][j] = map[0][j] = '|' ;
	}		

	/*  Create pthreads for wood move and frog control.  */
	for (int rank = 0; rank < NUM_LOGS; rank++){
		pthread_create(&threads[rank], NULL, logs_move, (void *)&thread_ids[rank]);
	}
	pthread_create(&p_frog, NULL, f_frog, NULL);

	/*  the main thread was used to fresh the screnn   */
	refresh_map(argc, argv);

	for (int rank = 0; rank < NUM_LOGS; rank++){
		pthread_join(threads[rank], NULL);
	}
	pthread_join(p_frog, NULL);

	/*  Display the output for user: win, lose or quit.  */
	printf("\033[2J\033[H");
	if (status == 0){
		printf("you lose the game!\n");
	}
	else if (status == 2){
		printf("you win the game!\n");
	}else{
		printf("you quit the game!\n");
	}
	pthread_exit(NULL);
	return 0;

}
