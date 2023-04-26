Troubleshooting
==============

VirtualBox
------

- Can only see */parameter_events* and */rosout* topics when ros2 topic list command is run:

  - Go to VirtualBox settings and Network and make sure the network type is Bridged instead of NAT.
  - Restart your virtual box.

Turtlebot4
-------
- Create3 topics suddenly disappeared:

  - Run the following command:
.. code-block:: console
    $ sudo systemctl stop unattended-upgrades
    $ sudo apt-get purge unattended-upgrades
  - Restart Turtlebot4

