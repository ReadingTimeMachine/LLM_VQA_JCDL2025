# All metrics should have form of:
# def metric( R1, R2, **params)
# with     
#   R1/R2 : [xmin, ymin, xmax, ymax]

from sys import path
path.append('../')
from metric_utils.utilities import isRectangleOverlap
import numpy as np

def IoU(R1, R2, return_individual = False, epsilon=1e-20, 
       check_rectangle_overlap = True): 
    '''
    Calculate IOU between box1 and box2

    Parameters
    ----------
    - R1 : [xmin, ymin, xmax, ymax]
    - R2 : [xmin, ymin, xmax, ymax]
    - return_individual: return intersection, union and IOU? default is False
    - check_rectangle_overlap : sets iou to zero if no overlap
    
    Returns
    -------
    - IOU
    '''   
    xmin1, ymin1, xmax1, ymax1 = R1
    xmin2, ymin2, xmax2, ymax2 = R2
    # box 1
    x1 = 0.5*(xmax1 + xmin1)
    y1 = 0.5*(ymax1 + ymin1)
    w1 = np.abs(xmax1 - xmin1)
    h1 = np.abs(ymax1 - ymin1)
    # box 2
    x2 = 0.5*(xmax2 + xmin2)
    y2 = 0.5*(ymax2 + ymin2)
    w2 = np.abs(xmax2 - xmin2)
    h2 = np.abs(ymax2 - ymin2)
    # calcs
    interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interx * intery
    union = w1*h1 + w2*h2 - inter
    iou = inter / (union + epsilon)
    if check_rectangle_overlap:
        if not isRectangleOverlap(R1,R2):
            iou = 0
            inter = 0
            union = w1*h1 + w2*h2 - inter
        #else:
        #    print('ok!')
    if not return_individual:
        return iou
    else:
        return inter, union, iou


def GIoU( R1, R2 ):
    """
    R1 : [xmin1, ymin1, xmax1, ymax1]
    R2 : [xmin2, ymin2, xmax2, ymax2]
    image : if set to tru assumes y = 0 at the top of the page
    """
    xmin1,ymin1,xmax1,ymax1 = R1
    xmin2,ymin2,xmax2,ymax2 = R2
    # calcs
    # interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    # intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    # w1, h1 = xmax1-xmin1, ymax1-ymin1
    # w2, h2 = xmax2-xmin2, ymax2-ymin2
    # inter = interx * intery
    # union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
    # iou = inter/(union + 1e-10) # so no divide by 0
    inter,union,iou = IoU(R1,R2,return_individual=True)
    wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2)
    hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2)
    C = np.abs(wc*hc)
    # put it all together
    #giou = 1 - iou + np.abs(C - union)/(C + 1e-10) # no divide by 0 --> this is in terms of the loss
    giou = iou - np.abs(C - union)/(C + 1e-10) # no divide by 0
    return giou


def DIoU( R1, R2, verbose=False):
    """
    R1 : [xmin1, ymin1, xmax1, ymax1]
    R2 : [xmin2, ymin2, xmax2, ymax2]
    """
    xmin1,ymin1,xmax1,ymax1 = R1
    xmin2,ymin2,xmax2,ymax2 = R2
    # calcs
    # interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    # intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    # w1, h1 = xmax1-xmin1, ymax1-ymin1
    # w2, h2 = xmax2-xmin2, ymax2-ymin2
    # inter = interx * intery
    # union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
    # iou = inter/(union + 1e-10) # so no divide by 0
    iou = IoU(R1,R2)
    wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2)
    hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2)
    #C = np.abs(wc*hc)
    # calculate distance between centers of boxes
    x1,y1 = 0.5*(xmin1+xmax1), 0.5*(ymin1+ymax1)
    x2,y2 = 0.5*(xmin2+xmax2), 0.5*(ymin2+ymax2)
    rho = np.sqrt( (x1-x2)**2.0 + (y1-y2)**2.0 )
    # calculate diagonal distance of C
    c = np.sqrt( wc**2.0 + hc**2.0 )
    # put it all together
    diou = iou - rho*rho/c**2
    if verbose:
        print(rho*rho/c**2)
    return diou


def CIoU( R1, R2, verbose=False):
    """
    R1 : [xmin1, ymin1, xmax1, ymax1]
    R2 : [xmin2, ymin2, xmax2, ymax2]
    """
    xmin1,ymin1,xmax1,ymax1 = R1
    xmin2,ymin2,xmax2,ymax2 = R2
    # calcs
    # interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    # intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    w1, h1 = xmax1-xmin1, ymax1-ymin1
    w2, h2 = xmax2-xmin2, ymax2-ymin2
    # inter = interx * intery
    # union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
    # iou = inter/(union + 1e-10) # so no divide by 0
    iou = IoU(R1,R2)
    # extra
    wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2)
    hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2)
    #C = np.abs(wc*hc)
    # calculate distance between centers of boxes
    x1,y1 = 0.5*(xmin1+xmax1), 0.5*(ymin1+ymax1)
    x2,y2 = 0.5*(xmin2+xmax2), 0.5*(ymin2+ymax2)
    rho = np.sqrt( (x1-x2)**2.0 + (y1-y2)**2.0 )
    # calculate diagonal distance of C
    c = np.sqrt( wc**2.0 + hc**2.0 )
    v = 4.0/np.pi**2*( np.arctan2(h1,w1) - np.arctan2(h2,w2) )**2.0
    alpha = v/( (1-iou) + v )
    # put it all together
    ciou = iou - rho*rho/c**2 - alpha*v
    return ciou



def EIoU( R1, R2, verbose=False):
    """
    R1 : [xmin1, ymin1, xmax1, ymax1]
    R2 : [xmin2, ymin2, xmax2, ymax2]
    """
    xmin1,ymin1,xmax1,ymax1 = R1
    xmin2,ymin2,xmax2,ymax2 = R2
    # calcs
    iou = IoU(R1,R2)
    # interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    # intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    w1, h1 = xmax1-xmin1, ymax1-ymin1
    w2, h2 = xmax2-xmin2, ymax2-ymin2
    # inter = interx * intery
    # union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
    # iou = inter/(union + 1e-10) # so no divide by 0
    # width and height of the minimum bounding box for both boxes
    wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2)
    hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2)
    #C = np.abs(wc*hc)
    # calculate distance between centers of boxes
    x1,y1 = 0.5*(xmin1+xmax1), 0.5*(ymin1+ymax1)
    x2,y2 = 0.5*(xmin2+xmax2), 0.5*(ymin2+ymax2)
    rho = np.sqrt( (x1-x2)**2.0 + (y1-y2)**2.0 )
    # calculate diagonal distance of C
    c = np.sqrt( wc**2.0 + hc**2.0 )
    # width term
    # here I'm assuming they also mean Euclidean distance for one point    
    rho_width = np.sqrt( (w1-w2)**2 ) 
    # height term
    # same deal with Eucledian distance
    rho_height = np.sqrt( (h1-h2)**2 ) 
    # put it all together
    eiou = iou - rho*rho/c**2 - rho_width**2/wc**2 - rho_height**2/hc**2
    return eiou


# # from: SIoU Loss: More Powerful Learning for Bounding Box Regression
# def SIoU( R1, R2, verbose=False, theta = 4):
#     """
#     R1 : [xmin1, ymin1, xmax1, ymax1]
#     R2 : [xmin2, ymin2, xmax2, ymax2]
#     theta : from the paper: The value of theta is a very important term in this equation, it controls how much attention should be paid to the cost of the shape. If value of theta is set to be 1, it will immediately optimize the shape, thus harm free movement of a shape. To calculate value of theta the genetic algorithm is used for each dataset, experimentally the value of theta near to 4 and the range that the author defined for this parameter is from 2 to 6.
#     """
#     xmin1,ymin1,xmax1,ymax1 = R1
#     xmin2,ymin2,xmax2,ymax2 = R2
#     # calcs
#     interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
#     intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
#     w1, h1 = xmax1-xmin1, ymax1-ymin1
#     w2, h2 = xmax2-xmin2, ymax2-ymin2
#     inter = interx * intery
#     union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
#     iou = inter/(union + 1e-10) # so no divide by 0
#     # width and height of the minimum bounding box for both boxes
#     wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2) # their c_w
#     hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2) # their c_h?
#     # centers of boxes
#     x1,y1 = 0.5*(xmin1+xmax1), 0.5*(ymin1+ymax1)
#     x2,y2 = 0.5*(xmin2+xmax2), 0.5*(ymin2+ymax2)
#     # for centers differences
#     c_h = np.maximum(y1,y2) - np.minimum(y1,y2)
#     c_w = np.maximum(x1,x2) - np.minimum(x1,x2)

#     # calculate sigma
#     sigma = np.sqrt( (x1-x2)**2.0 + (y1-y2)**2.0 )
#     print('sigma = ', sigma, 'hc =', hc, 'c_h=', c_h)
    
#     # calculate X
#     X = c_h/sigma
#     print('X = ', X, 'X/pi=', X/np.pi)
        
    
#     # calculate Lambda
#     Lambda = 1.0 - 2 * (np.sin( np.arcsin(X) - np.pi/4.0) )**2.0
#     print('arcsin(X) = ', np.arcsin(X))
#     print('Lambda = ', Lambda)
    
#     # calculate Delta
#     rho_x = ( (x1-x2)/c_w ) ** 2
#     rho_y = ( (y1-y2)/c_h ) ** 2
#     gamma = 2.0 - Lambda
#     Delta = (1 - np.e**(-1.0*gamma * rho_x)) + (1 - np.e**(-1.0*gamma * rho_y))

#     # calculate Omega
#     omega_w = np.abs(w1-w2)/np.maximum(w1,w2)
#     omega_h = np.abs(h1-h2)/np.maximum(h1,h2)
#     Omega = (1 - np.e**(-1.0*omega_w))**theta + (1 - np.e**(-1.0*omega_h))**theta
#     # put it all together!
#     siou = iou - (Delta + Omega)/2

#     return siou



# Focaler-IoU: More Focused Intersection over Union Loss
def SIoU( R1, R2, verbose=False, theta = 4, epsilon=1e-20):
    """
    R1 : [xmin1, ymin1, xmax1, ymax1]
    R2 : [xmin2, ymin2, xmax2, ymax2]
    theta : from the paper: The value of theta is a very important term in this equation, it controls how much attention should be paid to the cost of the shape. If value of theta is set to be 1, it will immediately optimize the shape, thus harm free movement of a shape. To calculate value of theta the genetic algorithm is used for each dataset, experimentally the value of theta near to 4 and the range that the author defined for this parameter is from 2 to 6.
    epsilon : a small number to avoid 1/0 types of issues
    """
    xmin1,ymin1,xmax1,ymax1 = R1
    xmin2,ymin2,xmax2,ymax2 = R2
    # calcs
    iou = IoU(R1,R2)
    # interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    # intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    w1, h1 = xmax1-xmin1, ymax1-ymin1
    w2, h2 = xmax2-xmin2, ymax2-ymin2
    # inter = interx * intery
    # union = np.abs(w1*h1) + np.abs(w2*h2) - np.abs(inter)
    # iou = inter/(union + 1e-10) # so no divide by 0
    # width and height of the minimum bounding box for both boxes
    wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2) 
    hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2) 
    # centers of boxes -- these are their x_c, y_c, etc
    x1,y1 = 0.5*(xmin1+xmax1), 0.5*(ymin1+ymax1)
    x2,y2 = 0.5*(xmin2+xmax2), 0.5*(ymin2+ymax2)

    # calculate Lambda
    X = np.minimum(np.abs(x1-x2), np.abs(y1-y2))/(np.sqrt( (x1-x2)**2 + (y1-y2)**2 ) + epsilon)
    Lambda = np.sin(2.0*np.arcsin(X))

    # calculate Omega
    omega_w = np.abs(w1-w2)/np.maximum(w1,w2)
    omega_h = np.abs(h1-h2)/np.maximum(h1,h2)
    Omega = (1.0 - np.e**(-1.0*omega_w))**theta + (1.0 - np.e**(-1.0*omega_h))**theta

    # calculate Delta
    rho_w = ( (x1-x2)/wc )**2.0
    rho_h = ( (y1-y2)/hc )**2.0
    gamma = 2.0 - Lambda
    Delta = (1.0 - np.e**(-1.0*gamma*rho_w)) + (1.0 - np.e**(-1.0*gamma*rho_h))
    
    # put it all together
    siou = iou - (Delta + Omega)/2.0
    return siou


# NOT USED -- requires hyperparameter tuning
# def BIoU(R1, R2, verbose=False):
#     """
#     R1 : [xmin1, ymin1, xmax1, ymax1]
#     R2 : [xmin2, ymin2, xmax2, ymax2]
#     theta : from the paper: The value of theta is a very important term in this equation, it controls how much attention should be paid to the cost of the shape. If value of theta is set to be 1, it will immediately optimize the shape, thus harm free movement of a shape. To calculate value of theta the genetic algorithm is used for each dataset, experimentally the value of theta near to 4 and the range that the author defined for this parameter is from 2 to 6.
#     epsilon : a small number to avoid 1/0 types of issues
#     """
#     xmin1,ymin1,xmax1,ymax1 = R1
#     xmin2,ymin2,xmax2,ymax2 = R2

#     iou = IoU(R1,R2) # get iou

#     wc = np.maximum(xmax1,xmax2) - np.minimum(xmin1,xmin2) 
#     hc = np.maximum(ymax1,ymax2) - np.minimum(ymin1,ymin2) 

#     # gamma
#     WC = gamma*wc
#     HC = gamma*hc
    
#     R = (WC + HC + MNE + MXE)/c**2
    
#     biou = IoU - R
    
#     return biou