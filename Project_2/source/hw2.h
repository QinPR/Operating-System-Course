#ifndef HW2_H
#define HW2_H

#include <QWidget>
#include <QPushButton>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QTimer>
#include <QLabel>
#include <QPaintEvent>
#include <QPainter>
#include <QKeyEvent>
#include <QSlider>

class MainWidget : public QWidget{
	Q_OBJECT
	public:
		explicit MainWidget(QWidget *parent = 0);
		void keyPressEvent(QKeyEvent *event); 
	public slots:
    	void printmap();
		void setspeedValue(int value);
	private:
		QLabel *frog_gui;
		QLabel *game_msg;
		QTimer timer;
		QLabel* log[9][20];
		QLabel *upperbound;
		QLabel *lowerbound;
		QSlider *slider;
		QLabel *speed_text;
};

#endif