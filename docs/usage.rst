Usage
=====

Quick Start
-------------
If you have the ROS2 setup on your Turtlebot4 or any other mobile platform running ROS2 and installed all the required python dependencies you can follow the instructions in this section to get your robot moving to the commands of the user through the webcam of your laptop.

**Step 1**

Clone the repository into your robot platform running ROS2.

.. code-block:: console

   $ git clone https://github.com/VishalNadig/hand-gesture-controlled-turtlebot4.git

**Step 2**

Copy the cobot3_ws folder found in hand-gesture-controlled-turtlebot4 into your home directory and change into the cobot3_ws directory with the following commands:

.. code-block:: console

   $ cp -r hand-gesture-controlled-turtlebot4/cobot3_ws ~/
   $ cd ~/cobot3_ws/
   
 
**Step 3**

Build your workspace by running the command below:

.. code-block:: console

   $ colcon build
   
**Step 4**

Initialize setup.bash file:

.. code-block:: console

   $ . install/setup.bash
 
**Step 5**

Run the ROS2 node:

.. code-block:: console

   $ ros2 run cobot drive


Training the Hand Gesture Recognition Model
-----

To train the model, run the 'RF2_Media_Pipeline_version - Sign_Gesture_Classification.ipynb' jupyter notebook. You need to change the data directory value in the Jupyer Notebook to the location of the data directory in your local computer.
