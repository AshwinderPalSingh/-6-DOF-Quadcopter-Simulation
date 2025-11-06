import cadquery as cq

# Parameters
arm_length = 50
arm_width = 10
arm_height = 5
motor_mount_diameter = 12
motor_mount_height = 5
center_hub_diameter = 30
center_hub_height = 10

# Create the center hub
hub = cq.Workplane("XY").cylinder(center_hub_height, center_hub_diameter / 2)

# Create one arm
arm = cq.Workplane("XY").box(arm_length, arm_width, arm_height)
motor_mount = cq.Workplane("XY").cylinder(motor_mount_height, motor_mount_diameter / 2)

# Position motor mount on the end of the arm
positioned_mount = motor_mount.translate((arm_length / 2, 0, (arm_height + motor_mount_height) / 2))
single_arm_assembly = arm.union(positioned_mount).translate((arm_length / 2, 0, 0))

# Create four arms by rotating
arm1 = single_arm_assembly.rotate((0, 0, 0), (0, 0, 1), 0)
arm2 = single_arm_assembly.rotate((0, 0, 0), (0, 0, 1), 90)
arm3 = single_arm_assembly.rotate((0, 0, 0), (0, 0, 1), 180)
arm4 = single_arm_assembly.rotate((0, 0, 0), (0, 0, 1), 270)

# Union all parts into the final frame
frame = hub.union(arm1).union(arm2).union(arm3).union(arm4)

# Export to STL
output_filename = "quad_frame.stl"
cq.exporters.export(frame, output_filename)

print(f"Successfully generated CAD file: {output_filename}")
print("You can now open this file in a 3D viewer (e.g., F3D, PrusaSlicer, Windows 3D Viewer).")

