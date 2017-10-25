import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import matplotlib.image as mpimg
try:
    import cmocean
    cmap = cmocean.cm.phase
except ImportError:
    cmap = plt.cm.jet 

class potts_model():
    """
    period potts model for Q spins
    """

    def __init__(self,grid_size,Q):
        # size of grid
        self.Nx = grid_size[0]
        self.Ny = grid_size[1]

        # number of distinct spin states
        self.Q = Q

        # total energy
        self.E = None

        # discrete spin states
        self.spin_states = np.zeros((self.Nx,self.Ny),dtype=np.int16)

        # maximum spin state jump to make in MC move
        self.maxjump = max([self.Q/2,2])

        # kBT/J
        self.reduced_temperature = 1e-1

        # for pause button
        self.continue_running = True

        # number of attempted MC jumps
        self.num_attempts = 0

        # maximum number of jump attempts
        self.max_num_attempts = 1e8

        # strength of magnetic field
        self.magnetic_strength = 0.0

        # direction of magnetic field
        self.magnetic_field = 1

        # frequency at which to redraw canvas
        self.redraw_interval = 1

        # maximum number of samples to wait before redrawing plot
        self.max_redraw_interval = 2e4
     
        # include feedback to tune jump size
        self.num_accepted = 0

        # number of jumps rejected
        self.num_rejected = 0

        # number of moves in which to collect statistics for feedback
        self.feedback_size = 1000

        # play/pause flag
        self.running = False
    def initialise_spin_states(self):
        # random initialise spins
        self.spin_states = np.random.randint(low=1,high=self.Q+1,size=(self.Nx,self.Ny))
    def initialise_energy(self):
        """
        calculate initial total energy
        """
        self.E = 0.0

        for ix in range(self.Nx):
            for iy in range(self.Ny):
                self.E += self.single_state_energy_contribution(location=[ix,iy],\
                        spin=self.spin_states[ix,iy])
        # have quadruple counted bonds
        self.E *= 0.25

    def fetch_neighbour_spins(self,location):
        """
        return list of spins for neighbouring sites
        """
        ix = location[0]
        iy = location[1]

        if ix==0:
            left = self.Nx-1
            right = ix + 1
        elif ix == self.Nx - 1:
            left = ix - 1
            right = 0
        else:
            left = ix - 1
            right = ix + 1
        if iy==0:
            top = iy + 1
            bottom = self.Ny - 1
        elif iy == self.Ny-1:
            top = 0
            bottom = iy-1
        else:
            top = iy + 1
            bottom = iy - 1
        return np.asarray([self.spin_states[left,iy],self.spin_states[right,iy],\
                self.spin_states[ix,top],self.spin_states[ix,bottom]],dtype=np.float32)

    def single_state_energy_contribution(self,location,spin):
        """
        calculate energy contribution of a single spin state with neighbours
        """

        # contribution from magnetic field
        ext_field_contribution = self.magnetic_contribution(spin)

        # retreive 4 neighbouring spin states
        neighbouring_spin = self.fetch_neighbour_spins(location)
        
        # theta = 2 pi n / Q , n=[1,Q]
        return -np.sum( np.cos( (neighbouring_spin - spin)*2*np.pi/self.Q ) ) + ext_field_contribution

    def magnetic_contribution(self,spin):
        """
        magnetic contribution to total energy of a single spin
        """
        return -self.magnetic_strength*np.cos(2.0*np.pi*(self.magnetic_field-spin)/self.Q)

    def attempt_single_jump(self):
        ix = np.random.randint(low=0,high=self.Nx)
        iy = np.random.randint(low=0,high=self.Ny)

        # current spin state for grid point
        current_spin = self.spin_states[ix,iy]

        # current energy contribution
        current_energy = self.single_state_energy_contribution([ix,iy],current_spin)

        # propose new spin state
        proposed_spin = current_spin + np.random.randint(low=-self.maxjump,high=self.maxjump)

        # map to [1,Q]
        proposed_spin = np.mod(proposed_spin,self.Q)
        if proposed_spin == 0:
            proposed_spin = self.Q

        # proposed energy contribution
        proposed_energy = self.single_state_energy_contribution([ix,iy],proposed_spin)

        # total energy change
        energy_change = proposed_energy - current_energy

        if energy_change < 0.0:
            # always accept negative energy changes
            accept = True
        elif np.exp(-energy_change/self.reduced_temperature) > np.random.uniform(low=0,high=1.0):
            # accept with probability exp( - energy_change / reduced_temperature )
            accept = True
        else:
            accept = False

        if accept:
            # update spin state
            self.spin_states[ix,iy] = proposed_spin

            # update total energy
            self.E += energy_change
        return accept

    def initialise(self):
        self.initialise_spin_states()
        self.initialise_energy()

    def play_pause(self,event):
        if self.running:
            self.running = False
        else:
            self.running = True

            while self.num_attempts < self.max_num_attempts and self.running:
                move_accepted = self.attempt_single_jump()

                self.num_attempts += 1

                # tune jump size so that # accepted / # rejected = 1
                if self.num_accepted + self.num_rejected < self.feedback_size:
                    if move_accepted:
                        self.num_accepted += 1
                    else:
                        self.num_rejected += 1
                else:
                    acceptance_frac = float(self.num_accepted)/float(self.num_rejected)
                    if acceptance_frac < 3.0/10.0:
                        self.maxjump += 1
                    elif acceptance_frac > 10.0/3.0:
                        self.maxjump -= 1

                    if move_accepted:
                        self.num_accepted = 1
                        self.num_rejected = 0
                    else:
                        self.num_accepted = 0
                        self.num_rejected = 0


                if np.mod(self.num_attempts,self.redraw_interval)==0:
                    # update num of MC steps
                    self.numsteps_ax.clear()
                    self.numsteps_ax.text(2,6,"number of MC steps : {}".\
                            format(self.num_attempts),fontsize=10,alpha=1)

                    self.replot_spinstates()
                    self.fig.canvas.flush_events()
                    self.fig.canvas.draw()
    
    def speedup(self,event):
        if self.redraw_interval*2 < self.max_redraw_interval:
            self.redraw_interval *= 2
    
    def slowdown(self,event):
        self.redraw_interval = int(max([1,self.redraw_interval/2]))

    def replot_spinstates(self):
        self.ax1.imshow(X=self.spin_states,vmin=0,vmax=self.Q,cmap=cmap)

    def view_grid(self,fig,ax1,numsteps_ax):
        self.fig = fig
        self.ax1 = ax1
        self.numsteps_ax = numsteps_ax
    
        # white figure background
        self.fig.patch.set_facecolor('white')

        self.ax1.set_aspect('equal')

        self.replot_spinstates()

if __name__ == "__main__":
    ex1 = potts_model(grid_size=[100,100],Q=10)
    ex1.initialise()

    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

    # white image background
    fig.patch.set_facecolor('white')
    
    # UC logo
    ax_logo = fig.add_axes([-0.225,0.87,0.7,0.12])
    ax_logo.imshow(mpimg.imread('logo.png'),alpha=1)
    ax_logo.set_aspect('equal')
    ax_logo.axis('off')

    # course name
    course_name = fig.add_axes([-0.59,-0.45,0.3,0.08])
    course_name.text(2,6,"course name",fontsize=10,alpha=0.6)
    course_name.axis('off')
    

    # MC steps
    numsteps_ax = fig.add_axes([-0.3,-0.45,0.3,0.08])
    numsteps_ax.text(2,6,"number of MC steps : {}".format(0),fontsize=10,alpha=1)
    numsteps_ax.axis('off')

    # attach figure to class
    ex1.view_grid(fig,ax1,numsteps_ax)


    # axes for buttons
    run_button = plt.axes([0.01,0.7,0.08,0.05])
    speedup_button = plt.axes([0.01,0.5,0.08,0.05])
    slowdown_button = plt.axes([0.01,0.3,0.08,0.05])

    # event buttons
    Brun = Button(run_button,u"\u25B6"+"||",color='0.9',hovercolor='0.99')
    Bspeedup = Button(speedup_button,'speed up',color='0.9',hovercolor='0.99')
    Bslowdown = Button(slowdown_button,'slow down',color='0.9',hovercolor='0.99')

    # event handlers
    Brun.on_clicked(ex1.play_pause)
    Bspeedup.on_clicked(ex1.speedup)
    Bslowdown.on_clicked(ex1.slowdown)

    plt.show()
