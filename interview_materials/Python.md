# Python 面试题

## 1、python的垃圾回收机制

Python中的垃圾回收是以引用计数为主，标记-清除和分代收集为辅。引用计数最大的缺陷就是循环引用的问题，所以python采用了辅助方法。详情参考（ https://www.cnblogs.com/yekushi-Z/p/11474211.html ）。

**注意：**

　　1、垃圾回收时，Python不能进行其它的任务，频繁的垃圾回收将大大降低Python的工作效率；

　　2、Python只会在特定条件下，自动启动垃圾回收（垃圾对象少就没必要回收）

　　3、当Python运行时，会记录其中分配对象(object allocation)和取消分配对象(object deallocation)的次数。当两者的差值高于某个阈值时，垃圾回收才会启动。

1. 引用计数

    引用计数法的原理是每个对象维护一个ob_refcnt，用来记录当前对象被引用的次数，也就是来追踪到底有多少引用指向了这个对象。当一个对象有新的引用时，它的ob_refcnt就会增加，当引用它的对象被删除，它的ob_refcnt就会减少。当引用计数为0时，该对象生命就结束了。 

   - 引用计数增加的情况：

   1. 对象被创建：x='spam'
   2. 用另一个别名被创建：y=x
   3. 被作为参数传递给函数：foo(x)
   4. 作为容器对象的一个元素：a=[1,x,'33']

   - 引用计数减少情况

   1. 一个本地引用离开了它的作用域。比如上面的foo(x)函数结束时，x指向的对象引用减1。
   2. 对象的别名被显式的销毁：del x ；或者del y
   3. 对象的一个别名被赋值给其他对象：x=789
   4. 对象从一个窗口对象中移除：myList.remove(x)
   5. 窗口对象本身被销毁：del myList，或者窗口对象本身离开了作用域

2. 标记-清除算法

   『标记清除（Mark—Sweep）』算法是一种基于追踪回收（tracing GC）技术实现的垃圾回收算法。它分为两个阶段：第一阶段是标记阶段，GC会把所有的『活动对象』打上标记，第二阶段是把那些没有标记的对象『非活动对象』进行回收。那么GC又是如何判断哪些是活动对象哪些是非活动对象的呢？

   对象之间通过引用（指针）连在一起，构成一个有向图，对象构成这个有向图的节点，而引用关系构成这个有向图的边。从根对象（root object）出发，沿着有向边遍历对象，可达的（reachable）对象标记为活动对象，不可达的对象就是要被清除的非活动对象。根对象就是全局变量、调用栈、寄存器。

   标记清除算法作为Python的辅助垃圾收集技术主要处理的是一些容器对象，比如list、dict、tuple，instance等，因为对于字符串、数值对象是不可能造成循环引用问题。Python使用一个双向链表将这些容器对象组织起来。不过，这种简单粗暴的标记清除算法也有明显的缺点：清除非活动的对象前它必须顺序扫描整个堆内存，哪怕只剩下小部分活动对象也要扫描所有对象。

3. 分代回收算法

   - Python将所有的对象分为0，1，2三代；
   - 所有的新建对象都是0代对象；
   - 当某一代对象经历过垃圾回收，依然存活，就被归入下一代对象。

   python在创建对象之前，会创建一个链表，零代链表，只不过这个链表是空的。每当你创建一个对象，python便会将其加入到零代链表。

    

   python隔代回收的核心：对链子上的那些明明没有被引用但引用计数却不是零的对象进行引用计数减去一，看看你是不是垃圾。如果被引用多次减去一之后仍不为零，那么会在零代链表当中继续被清理，直至引用计数为零。因为如果没有变量指向它，或者作为函数的参数，列表的元素等等，那么它就始终是零代链表中被清理的对象。当零代链表被清理达到一定次数时，会清理一代链表。一代链表被清理达到一定次数时，会清理二代链表。

   因此清理的频率最高的是零代链表，其次是一代链表，再是二代链表。

**注：python内存管理机制**

**原理：**

1. Python提供了对内存的垃圾收集机制，但是它将不用的内存放到内存池而不是返回给操作系统；
2. Pymalloc机制：为了加速Python的执行效率，Python引入了一个内存池机制，用于管理对小块内存的申请和释放；
3. 对于Python对象，如整数，浮点数和List，都有其独立的私有内存池，对象间不共享他们的内存池。也就是说如果你分配又释放了大量的整数，用于缓存这些整数的内存就不能再分配给浮点数。



## 2、python装饰器

-  **2.1 什么是装饰器？**

  装饰器即函数的函数，因为装饰器传入的参数就是一个函数，然后通过实现各种功能来对这个函数的功能进行增强。

  装饰器其实就是一个闭包，把一个函数当做参数后返回一个替代版函数，闭包是装饰器的核心。 

- **2.2 为什么要用装饰器？**

  使用方便，只需要在函数上方加一个@就可以对其进行增强。使代码具有python简洁的风格。

- **2.3 在哪里用装饰器？**

  装饰器最大的优势是用于解决重复性的操作，主要使用场景有如下几个：

  - 计算函数的运行时间；
  - 给函数打印日志
  - 类型检查。

  当然，如果遇到其他重复的场景也可以使用装饰器。

- **2.4 保留元信息的装饰器？**

  函数的元信息就是函数携带的一些基本信息，如函数名、函数文档等。由于使用装饰器，函数的元信息会丢失，打印出来的是装饰器包装后的函数的信息。

  可以from functiontools import wraps, 在def wrapper()函数上方加上@wraps(func)即可保留原函数的元信息。

- **2.5 扩展：简单解释下闭包的特点：**

  一个函数返回的函数对象，这个函数对象执行的话依赖非函数内部的变量值，这个时候，函数返回的实际内容如下：
  1 函数对象
  2 函数对象需要使用的外部变量和变量值
  **以上就是闭包**，**闭包必须嵌套在一个函数里，必须返回一个调用外部变量的函数对象，才是闭包**

## 3、*args和**kargs的用法

- **问题引入：**

  定义一个简单函数：

  ```python
  def jiafa(x,y):
      z = x + y
      return z
  print(jiafa(1,2))
  ```

  上述定义的函数是固定个数的参数的加和。x,y是位置参数，是固定的，是按照位置顺序传参。如果参数个数是不固定的又该如何定义函数参数呢？这就用到了*args。

- ***args的用法：**

  python中规定参数前带有*的，称为**可变位置参数**.

  通常称这个可变位置参数为*args。

  ```python
  def jiafa(*args):
      sum = 0
      for i in args:
          sum = sum + i
      return sum
  print(jiafa(1,3,5))
  print(jiafa(2,4,6,8,10,))
  ```

  

- **kargs的用法：

  python中规定参数前带有**的，称为可变关键字参数。

  通常用**kargs表示。

  ```python
  def zidian(**kargs):
      print(kargs)
  zidian(a=1,b=2,c=3)
  zidian(a=1,b=2,c=3,d=4)
  ```

  

## 4、python的基本数据类型有哪些？

- （1）Number（数字）

  python3中数字类型又包含int、long、float、complex（复数）、bool（布尔值，只有True和False）

- （2）String（字符串）

  有三种不同的字符串类型：

  - “unicode”:表示Unicode字符串（文本字符串）
  - “str”:表示字节字符串（二进制数据）
  - “basestring”：表示前两种字符串的父类。

- （3）List（列表）

- （4）Tuple（元组）

  与列表类似，不同之处在于：

  - **元组的元素不能修改**。
  - **元组使用小括号，而列表使用方括号**

- （5）Set（集合）

- （6）Dictionary（字典）

总结：

上述数据类型，Number和String是基本数据类型，而List、Tuple、Set、Dictionary是容器类型。