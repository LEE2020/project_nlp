#!/bin/env/python
# -*- encoding:utf-8 -*-
/**********************************************************
 * Author        : 07zhiping 
 * Email         : 07zhiping@gmail.com 
 * Last modified : 2020-10-31 13:46
 * Filename      : trans.py
 * Description   : 
 * *******************************************************/

class queen:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self,x):
        self.stack1.append(x)
        
    def pop(self):
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        rst = self.stack2.pop()
        while self.stack2:
            self.stack2.append(self.stack2.pop())
        return rst         

