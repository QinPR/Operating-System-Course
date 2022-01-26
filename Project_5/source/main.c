#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"


MODULE_LICENSE("GPL");

static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;


#define PREFIX_TITLE "OS_AS5"
#define IRQ_NUM 1


// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );


static int count = 0;
/* the functions for interrupts */
static irqreturn_t irq_handler(int data, void *dev_id)
{
	count += 1;
    return IRQ_HANDLED;
}



// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};


// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	int readable;
	readable = myini(DMAREADABLEADDR);
	while (readable == 0){
		msleep(1);
		readable = myini(DMAREADABLEADDR);      /* busy waiting there if not readable */
	}    
	int ans;
	ans = myini(DMAANSADDR);
	put_user(ans, (int*)buffer);
	myouti(0, DMAREADABLEADDR);                /* once read the answer, set the reaable to 0 */

	printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, ans);
	return 0;
}
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	int IOMode;
	IOMode = myini(DMABLOCKADDR);
 	get_user(dataIn->a, (char*)buffer);
	get_user(dataIn->b, (int*)(buffer + 4));
	get_user(dataIn->c, (short*)(buffer + 8));
	myoutc(dataIn->a, DMAOPCODEADDR);
	myouti(dataIn->b, DMAOPERANDBADDR);
	myouti(dataIn->c, DMAOPERANDCADDR);
	INIT_WORK(work_routine, drv_arithmetic_routine);


	// Decide io mode
	if(IOMode) {      /* Blocking IO */
		printk("%s,%s(): queue work\n",PREFIX_TITLE, __func__);
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();
    	} 
	else {           /* Non-locking IO */
		printk("%s,%s(): queue work\n",PREFIX_TITLE, __func__);
		schedule_work(work_routine);
   	}
	return 0;
}
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	int value;
 	get_user(value, (int *) arg);

	if (cmd == HW5_IOCSETSTUID){
		myouti(value, DMASTUIDADDR);
		if (value >= 0) printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, value);   /* student id cannot be negtive */
		else return -1;
	}
	if (cmd == HW5_IOCSETRWOK){
		myouti(value, DMARWOKADDR);
		if (value == 1) printk("%s:%s(): RM OK\n", PREFIX_TITLE, __func__);
		else return -1;
	}
	if (cmd == HW5_IOCSETIOCOK){
		myouti(value, DMAIOCOKADDR);
		if (value == 1) printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
		else return -1;
	}
	if (cmd == HW5_IOCSETBLOCK){
		myouti(value, DMABLOCKADDR);
		if (value == 1) printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
		else if (value == 0) printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		else return -1;
	} 
	if (cmd == HW5_IOCSETIRQOK){
		myouti(value, DMAIRQOKADDR);
		if (value == 1) printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
		else return -1;
	}
	if (cmd == HW5_IOCWAITREADABLE){
		int readable = myini(DMAREADABLEADDR);     /* check whether it is readable */
		printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
		while (readable != 1){
			msleep(1);
			readable = myini(DMAREADABLEADDR);
		}
		put_user(readable, (int *)arg);
		if(readable == 1) return 1;
		else return -1;
	}
	return 0;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
	char operator;
	int op1;
	int op2;
	operator = myinc(DMAOPCODEADDR);
	op1 = myini(DMAOPERANDBADDR);
	op2 = myini(DMAOPERANDCADDR);
	int num;                    /* the calculated result */
	if (operator == 'p'){
		int fnd=0;
		int i, isPrime;
		num = op1;
		while(fnd != op2) {
			isPrime=1;
			num++;
			for(i=2;i<=num/2;i++) {
				if(num%i == 0) {
					isPrime=0;
					break;
				}
			}
			if(isPrime) {
				fnd++;
			}
		}
	}
	else if (operator == '+'){
		num = op1 + op2;
	}
	else if (operator == '-'){
		num = op1 - op2;
	}
	else if (operator == '*'){
		num = op1 * op2;
	}
	else if (operator == '/'){
		num = op1 / op2;
	}
	myouti(num, DMAANSADDR);
	myouti(1, DMAREADABLEADDR);    /* when write finish, update it to readable */		
	printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__, op1, operator, op2, num);
}

static int __init init_modules(void) {
    
	/* Register chrdev */ 
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
	dev_t dev;
	int ret = 0;
	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	if(ret)
	{
		printk("Cannot alloc chrdev\n");
		return ret;
	}

	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("OS_AS5:init_modules(): register chrdev(%d,%d)\n",dev_major,dev_minor);

	/* Init cdev and make it alive */
	dev_cdevp = cdev_alloc();
	cdev_init(dev_cdevp, &fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
	if(ret < 0)
	{
		printk("Add chrdev failed\n");
		return ret;
	}

	/* add an isr */
	int result;
	result=request_irq(IRQ_NUM, (irq_handler_t) irq_handler, IRQF_SHARED, "Iterrupt_Counter_Device", &dev_major);
    printk("OS_AS5:init_modules(): request_irq %d return %d\n", IRQ_NUM, result);

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("OS_AS5:init_modules(): allocate dma buffer\n");

	dataIn = kmalloc(12, GFP_KERNEL);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {

	dev_t dev;
	
	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdevp);

	/* remove the isr */
	free_irq(IRQ_NUM, &dev_major);
	printk("OS_AS5:exit_modules(): interrupt count = %d\n", count);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("OS_AS5:exit_modules(): free dma buffer\n");

	unregister_chrdev_region(dev, 1);
	printk("OS_AS5:exit_modules(): unregister chrdev\n");

	/* free the work-routine */
	kfree(work_routine);

	kfree(dataIn);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
