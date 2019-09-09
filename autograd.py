#!/usr/bin/env python
# coding: utf-8

# In[148]:


import math
import numpy as np
import copy
class Differentiable:
    def __init__(self,*vars):
        self.vars = vars
    def __call__(self,x):
        raise NotImplementedError('Override this')
    def d(self,var):
        return (self.x.d(var) for x in self.vars)
    
    def __add__(self,other):
#         print(type(self),type(other))
        if type(self) == Constant and type(other) == Constant:
            return Constant(self.var+other.var)
        
        return Add(self,other)
    
    def __mul__(self,other):
#         if type(self) == Constant :
#             if self == Constant(0):
#                 return Constant(0)
#         elif type(other) == Constant:
#             if other == Constant(0):
#                 return Constant(0)
        
        
        return Multiply(self,other)
        
    def __sub__(self,other):
        return Subtract(self,other)
   
    def __truediv__(self,other):
        return Divide(self,other)
    def __pow__(self,other):
        return Pow(self,other)
    
    
    def  eval(self):
        raise NotImplementedError("Override This")
    
class Constant(Differentiable): 
    def __init__(self,val,name=None):
        self.val = val
        self.name = name
        
    def __str__(self):
        return f"{self.val}"
    
    def d(self,var):
        return Constant(0)
    
    def __eq__(self,other):
        if type(self) == type(other) and type(other) == Constant:
            return self.val == other.val 
        else:
            return False
    
    def __call__(self,x):
        return x
    
    def eval(self):
        return copy.deepcopy(self.val)
        
class Symbol(Differentiable):
    
    def __init__(self,name,val=None):
        self.val = val
        self.name = name
        
    def __str__(self):
        return f"{self.name}"
    
#     def __eq__(self,other):
#         if type(self) == type(other) and type(other) == Constant:
#             return self.name == other.name
#         else:
#             return False
    
    def d(self,var):
        if self == var:
            return Constant(1)
        else :
            return Constant(0)
    
    def __call__(self,x):
        return x
    
    def __repr__(self):
        return f"Name={self.name},val={self.val}"
    
    def eval(self):
        return copy.deepcopy(self.val)
    


# In[195]:



       
        

class Add(Differentiable):
    
    def __init__(self,*vars):
        super().__init__(*vars)
    
    def d(self,var):
        ds = [inte.d(var) for inte in self.vars]
        return Add(*ds)
    
    def __str__(self):
        rep = "("
        for inte in self.vars[:-1]:
            rep += f"{str(inte)} + "
        rep += f"{str(self.vars[-1])})"
        return rep
    
    def __call__(self,data):
        raise NotImplementedError("not made yet")

    def eval(self):
        ans = self.vars[0].eval()
        for t in self.vars[1:]:
            ans+=t.eval()
        
        return ans

class Subtract(Differentiable):
    
    def __init__(self,*vars):
        super().__init__(*vars)
    
    def d(self,var):
        ds = [inte.d(var) for inte in self.vars]
        return Subtract(*ds)
    
    def __str__(self):
        rep = "("
        for inte in self.vars[:-1]:
            rep += f"{str(inte)} - "
        rep += f"{str(self.vars[-1])})"
        return rep
    
    def __call__(self,data):
        raise NotImplementedError("not made yet")   
    
    
    def eval(self):
        ans = self.vars[0].eval()
        for t in self.vars[1:]:
            ans-=t.eval()
        
        return ans
    
    
class Multiply(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 2
        super().__init__(*vars)
    
    def d(self,var):
        
        u = self.vars[0]
        v = self.vars[1]
        
        return u.d(var) * v + v.d(var) * u 
        
        
    
    def __str__(self):
        rep = "("
        for inte in self.vars[:-1]:
            rep += f"{str(inte)} * "
        rep += f"{str(self.vars[-1])})"
        return rep
    
    def __call__(self,data):
        raise NotImplementedError("not made yet")
    
    
    def eval(self):
        ans = self.vars[0].eval()
        for t in self.vars[1:]:
            ans *= t.eval()
            
        return ans
    
class Divide(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 2
        super().__init__(*vars)
    
    def d(self,var):
        
        u = self.vars[0]
        v = self.vars[1]
        
        return (u.d(var) * v - v.d(var) * u)/v*v 
        
        
    
    def __str__(self):
        rep = "("
        for inte in self.vars[:-1]:
            rep += f"{str(inte)} / "
        rep += f"{str(self.vars[-1])})"
        return rep
    
    def __call__(self,data):
        raise NotImplementedError("not made yet")
    
    def eval(self):
        ans = self.vars[0].eval()
        
        for t in self.vars[1:]:
            ans /= t.eval()
        return ans
        

        
        
class Pow(Differentiable):
    def __init__(self,*vars):
        assert len(vars) == 2
        super().__init__(*vars)
    
    def d(self,var):
        base = self.vars[0]
        expo = self.vars[1]
        return (expo)*Pow(base,Constant(expo.val - 1))
    
    def __str__(self):
        return f"({self.vars[0]}**{self.vars[1]})"
    
    
    def eval(self):
        
        ans = self.vars[0].eval()**(self.vars[1].eval())
        return ans
        
#  --------------------------------------------------------------------------


class MatMul(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 2
        
        super().__init__(*vars)
    
    def d(self,var):
        return MatMul(self.vars[0],self.vars[1].d(var)) + MatMul(self.vars[0].d(var),self.vars[1])
    
    def __str__(self):
        
        return f"MatMul({str(self.vars[0])},{str(self.vars[1])})"
    
    def eval(self):
        return np.dot(self.vars[0].eval(),self.vars[1].eval())



class Relu(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 1
        super().__init__(*vars)
    
    def d(self,var):
        mask = var.eval() > 0
        mask = mask.astype(var.eval().dtype)
        return Constant(mask,"dRelu")*self.vars[0].d(var)
        
    def eval(self):
        ans = self.vars[0].eval()
        ans[ans < 0] = 0
        return ans
    
    def __str__(self):
        return f"Relu({str(self.vars[0])})"

    
    
class Sin(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 1
        super().__init__(*vars)
    
    def d(self,var):
        
        internal = self.vars[0]
        
        return Cos(*self.vars) * internal.d(var)
    
    def __str__(self,):
        rep = ""
        for var in self.vars:
            rep += f"{str(var)},"

        return f"Sin({rep})"

    def __call__(self,data):
        return math.sin(data)
     

class Cos(Differentiable):
    
    def __init__(self,*vars):
        assert len(vars) == 1
        super().__init__(*vars)
    
    def d(self,var):
        internal = self.vars[0]
        
        return Constant(-1)*Sin(*self.vars) * internal.d(var)
    
    def __str__(self,):
        rep = ""
        for var in self.vars:
            rep += f"{str(var)},"

        return f"Cos({rep})"
    
    def __call__(self,data):
        return math.cos(data)



    


# In[ ]:





# In[247]:


class Linear:
    def __init__(self,X:Constant):
        self.W = Symbol("W",0.4)
        self.b = Symbol("b",0)
        self.f = self.W*X+self.b
    
    def grads(self):
        return self.f.d(self.W),self.f.d(self.b)
        


# In[248]:


x = np.linspace(1,100,50)


# In[249]:


ys = 10 * x +89.23


# In[268]:


x = Constant(x)
y = Constant(ys)

model =Linear(x)


# In[269]:


loss = (model.f - y)**Constant(2)


# f.eval()

# In[270]:


print(loss.d(W))


# In[271]:


for epoch in range(10):
    print("Epoch 1")
    l = loss.eval()
    dW = loss.d(model.W).eval()
    db = loss.d(model.b).eval()
    model.W.val += 0.1*dW
    model.b.val += 0.1*db
    print(sum(l))


# In[272]:


print(loss)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




