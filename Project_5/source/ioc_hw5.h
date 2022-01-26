#ifndef IOC_HW5_H
#define IOC_HW5_H

#define HW5_IOC_MAGIC         'k'
#define HW5_IOCSETSTUID       _IOW(HW5_IOC_MAGIC, 1, int)    /* function to print out the student id */
#define HW5_IOCSETRWOK        _IOW(HW5_IOC_MAGIC, 2, int)    /* printk OK if complete R/W function */
#define HW5_IOCSETIOCOK       _IOW(HW5_IOC_MAGIC, 3, int)    /* printk OK if complete ioctl function */
#define HW5_IOCSETIRQOK       _IOW(HW5_IOC_MAGIC, 4, int)    /* printk OK if complete bonus */
#define HW5_IOCSETBLOCK       _IOW(HW5_IOC_MAGIC, 5, int)    /* set write function mode */
#define HW5_IOCWAITREADABLE   _IOR(HW5_IOC_MAGIC, 6, int)    /* used before read to confirm it can read answer now when use non-blocking write mode */
#define HW5_IOC_MAXNR         6

#endif




