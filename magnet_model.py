import torch
import torch.nn as nn
import torch.optim as opt

import numpy as np

#set the default tensor type as double
torch.set_default_tensor_type(torch.DoubleTensor)



#the constant PI
PI = np.pi


#the permability of free space
MU = 4 * PI * 1e-7



class MagnetArray(nn.Module):
  
  def __init__(self, N, min_radius, length, cube_magnet_size):
    
    """
    N : (int) number of magnets to have in the array
    min_radius : (float) the minimum radius that the magnets may be from the center of the ring
    length : (float) the side length of the cube magnet
    cube_magnet_size : (float) the size length of the cube magnets to be used in the array
    """
    
    super().__init__()
    
    
    self.num_magnets = N
    self.radius = min_radius
    self.length = length
    self.cube_magnet_size = cube_magnet_size
    
    #put the magnets on a cylinder with the minimum radius
    #align the magnet along the z axis
    #shape (N, 3)
    
    self.locations = 2*torch.rand((self.num_magnets, 3)) - 1
    
    self.locations[self.locations == 0.0] = 1e-12
    
    self.locations[:, 0:2] = self.radius * nn.functional.normalize(self.locations[:, 0:2], p=2, dim=1)
    
    self.locations[:, 2] = self.locations[:, 2] * length / 2
    
    #make the magnet locations a trainable parameter
    self.locations = nn.Parameter(self.locations)
    
    
    #randomly initialize the directions
    #shape (N, 3)
    self.directions = 2*torch.rand((self.num_magnets, 3)) - 1
    self.directions[self.directions == 0] = 1e-12
    
    #make the magnet directions a trainable parameter
    self.directions = nn.Parameter(self.directions)
    
    
  def display_magnets(self):
    
    #the magnet locations
    part1 = """
    positions = {0};
    
    orientations = {1};
    """.format(list_to_string(self.locations.tolist()),
               list_to_string(self.directions.tolist()))
    
    
    #the code to display the magnet locations
    part2 = """
    module point(p){

	x = p[0];
	y = p[1];
	z = p[2];

	R = norm([x, y]);

	theta = atan2(y, x);

	alpha = atan2(z, R);

	rotate([0, 0, theta])
	rotate([0, -alpha, 0])
	children();
      }
    
    for(i = [0:len(positions)-1]){
    
    translate(positions[i])
    point(orientations[i])
    linear_extrude(height=0.1, center=true, scale=1/2)
    square([0.05, 0.05], center=true);
    
    }"""
    
    
    
    
    
    return (part1, part2)
    
  
  def forward(self, X):
    """
    X : a pytorch tensor with shape (n, 3) where n is the number of samples
    this represents the points in space to sample the field
    
    returns the field at the sampled points
    a pytorch tensor with shape (n, 3)
    """
    
    global MU, PI
    
    
    #find the vectors pointing from each magnet to each sample position
    #shape (n, N, 3)
    distance_vectors = X.unsqueeze(1).repeat((1, self.num_magnets, 1))
    distance_vectors = self.locations.unsqueeze(0) - distance_vectors
    
    #name to match the equations
    g = distance_vectors
    
    
    #calculate the distances
    #shape (n, N, 1)
    distances = torch.sqrt((distance_vectors**2).sum(dim=2, keepdims=True))
    
    #name to match the equations
    d = distances
    
    #dipole moment for each magnet
    #shape (1, N, 3)
    m = (self.cube_magnet_size**3) * nn.functional.normalize(self.directions, dim=1).unsqueeze(0)
    bRem = 1.3
    m = bRem * m / MU
    
    #shape (n, N, 3)
    part1 = 3*g*((m*g).sum(dim=2, keepdims=True)) / (d**5)
    
    #shape (1, N, 3)
    part2 = m / (d**3)
    
    #the magnetic field vector at each sample
    #shape (n, N, 3)
    B = MU / (4*PI)
    B = B * (part1 - part2)
    
    
    
    #sum up the various magnetic influences at the different points
    #shape (n, 3)
    B = B.sum(dim=1)
    
    
    
    return B
    
    
    


class Homogeneity(nn.Module):
  
  def __init__(self):
    super().__init__()
    
  def forward(self, X, multiplier=1e6):
    """
    this function calculates the homogeneity of the magnetic field
    X is a pytorch tensor with shape (n, 3) where n is the number of points sampled in the field
    """
    
    #field strength at the sampled points
    #shape (n)
    strengths = torch.sqrt((X**2).sum(dim=1))
    
    #homogeneity
    #shape (1), ie a scalar
    W = multiplier * (strengths.max() - strengths.min())/(strengths.mean() + 1e-12)
    
    return W
    
    
    

def list_to_string(lst):
  """
  converts a list of points to a string
  the list is expected to have format
  [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]]
  essentially with shape (n, 3)
  where n is the number of points in the list
  """
  string = "["
  for i in range(len(lst)-1):
    string += str(lst[i])
    string += ",\n"
  
  string += str(lst[len(lst)-1])
  
  string += "]"
  
  return string



def full_cylinder(num_points, r=1, h=1):
  """
  generates 3d samples uniformly from a cylinder of radius r and height h
  
  
  num_points : (int) the number of points to sample in from the cylinder
  r : (float) the radius of the cylinder
  h : (float) the height of the cylinder
  
  returns a pytorch tensor with shape (num_points, 3)
  """
  pi = 3.1415926535
  
  #shape (num_points, 2)
  coords = torch.rand((num_points, 2))
  
  coords[:, 0] = 2*pi * coords[:, 0]
  
  #shape (num_points, 3)
  points = torch.zeros(num_points, 3)
  
  #sample in a circle of radius r
  points[:, 0] = r*torch.sqrt(coords[:, 1])*torch.cos(coords[:, 0])
  
  points[:, 1] = r*torch.sqrt(coords[:, 1])*torch.sin(coords[:, 0])
  
  #sample in along the height of the cylinder
  points[:, 2] = (2*torch.rand(num_points)-1)*h/2
  
  return points  



def toScad(Arr=None):
  """
  outputs code to display the magnets and the field in OpenSCAD
  """
  
  if Arr is None:
    Arr = MagnetArray(320, 1/2, 1, 21/1000)
    
    #with torch.no_grad():
      #Arr.locations[0, :] = torch.Tensor([0, 0, 0])
      #Arr.directions[0, :] = torch.Tensor([1, 0, 0])
    
    with torch.no_grad():
      Arr.directions[:, :] = 0 
      Arr.directions += torch.Tensor([[0, 0, 1]])
  
  a = torch.linspace(-1, 1, 20)
  
  space = torch.cartesian_prod(a, a, a)
  
  field = Arr(space)
  
  #code for the field data
  data = """
  locations = {0};
  
  directions = {1};
  """.format(list_to_string(space.tolist()), list_to_string(field.tolist()))
  
  #code for displaying the field
  functions = """
  norms = [for(dir = directions) norm(dir)];

  max_norm = max(norms);

  min_norm = min(norms);
  
  for(i = [0:len(locations)-1]){
  
        strength = pow((norms[i] - min_norm) / (max_norm-min_norm), 1/2);

  
	color([0.4, 1-strength, 0.6, 1])
	
	hull(){
	translate(locations[i])
	cube(size=0.01, center=true);

	translate(locations[i] + 0.2*directions[i]/(norm(directions[i])+1e-12))
	cube(size=0.001, center=true);
  
  
  }
  
  }
  
  """
  
  p1, p2 = Arr.display_magnets()
  
  print(data + "\n" + p1 + "\n" + functions + "\n" + p2)










def optimize(epochs, 
             num_samples, 
             num_magnets=320, 
             min_radius=1/2, 
             length=1, 
             cube_magnet_size=21/1000):
  """
  optimize the magnet array to find the desired field characteristics
  
  epochs : (int) the number of training iterations to perform
  num_samples : (int) the number of point samples to test on each iteration
  num_magnets : (int) the number of magnets to have in the array
  min_radius : (float) the minimum radius that the magnets should stay out of in meters
  (does not work yet)
  length : (float) the length of the field in meters (along the z axis)
  cube_magnet_size : (float) the side length of the cube magnet in meters
  
  returns the trained MagnetArray object
  """
  
  #make the model
  Arr = MagnetArray(num_magnets, min_radius, length, cube_magnet_size)
  
  Hom = Homogeneity()

  #training parameters
  rate = 1
  momentum = 0.9
  clip = 0.01
  
  #loss function and optimizer
  loss_function = nn.MSELoss()
  optimizer = opt.SGD(Arr.parameters(),
                      lr=rate,
                      momentum=momentum)
  

  #optimizer = opt.Adam(Arr.parameters(), lr=0.001, betas=(0.9, 0.999))
  
  #printing setting
  prynt = 200
  recalculate = 10
  running_loss = 0
  
  #dev = torch.device("cuda:0")
  #dev = torch.device("cpu")
  #Arr.to(dev)
  #Arr.train()
  
  #the target is set at zero
  y = torch.Tensor([0.0])
  
  for i in range(epochs):
    
    #resample the target points every so often
    if i % recalculate == 0:
      x = full_cylinder(num_samples, r=Arr.radius, h=Arr.length)
      
    
    Y = Hom(Arr(x))

    #backpropogate the loss
    loss = loss_function(Y, y)
    loss.backward() 
    
    #clip the gradient to avoid potential numerical instability
    nn.utils.clip_grad_norm_(Arr.parameters(), clip, norm_type=2)

    #adjust the weights
    optimizer.step()
  
    #print the training info
    running_loss += loss.item()
    if i % prynt == prynt-1:
        print('[%5d] loss: %.8f' % (i + 1, running_loss / prynt))
        running_loss = 0.0  
        
  return Arr
     
  
#train and print the trained model and the field
array_of_magnets = optimize(100_000, 200)
toScad(array_of_magnets)
