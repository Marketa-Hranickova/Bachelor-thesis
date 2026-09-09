// ADAPTER_PLATE.scad
// Adjustable-offset adapter between the chassis plate and ROAD_TRAIN tail joint.
// Moving the plate by one chassis hole changes d0 (or d1) by G_PITCH.
//
// Joint interface measured from ROAD_TRAIN_ASM_-_Tail_Joint_1/2.stl:
//   4 holes of 2.00 mm, rectangle 22.00 mm (across) x 8.00 mm (along)
//   pin axis sits 29.21 mm (Joint 1) / 34.21 mm (Joint 2) from that block
//
// BEFORE PRINTING, verify the four values marked TODO on the real robot.

// ---------------- plate ----------------
T  = 4.0;      // thickness [mm] -- 4 mm is stiff enough in-plane; go 5 if it flexes
W  = 30.0;     // width, across the vehicle
L  = 90.0;     // length, along the vehicle -- sets the adjustment range
CR = 4.0;      // corner radius

// ------------- joint interface (measured, do not change without re-measuring) ----
J_X      = 14.0;   // centre of the joint pattern, from the front edge
J_ALONG  =  8.0;   // hole spacing along the vehicle   (measured 8.00)
J_ACROSS = 22.0;   // hole spacing across the vehicle  (measured 22.00)
J_D      =  2.4;   // hole diameter: 2.00 nominal + clearance

// ------------- chassis interface ----------------
G_START  = 32.0;   // first chassis hole, from the front edge
G_PITCH  =  8.0;   // TODO verify: grid pitch of the chassis plate
G_N      =    7;   // number of positions -> range = (G_N-1)*G_PITCH = 48 mm
G_ACROSS = 20.0;   // TODO verify: transverse spacing of the chassis holes
G_D      =  3.4;   // TODO verify: M3 clearance

// Print a second copy with J_X = 14.0 + G_PITCH/2 to get half-pitch resolution.

$fn = 48;

module plate_body() {
    hull() for (x = [CR, L-CR], y = [CR, W-CR])
        translate([x, y, 0]) cylinder(h = T, r = CR);
}

module holes() {
    // joint pattern
    for (sx = [-1, 1], sy = [-1, 1])
        translate([J_X + sx*J_ALONG/2, W/2 + sy*J_ACROSS/2, -1])
            cylinder(h = T + 2, d = J_D);
    // chassis rows
    for (i = [0 : G_N-1], sy = [-1, 1])
        translate([G_START + i*G_PITCH, W/2 + sy*G_ACROSS/2, -1])
            cylinder(h = T + 2, d = G_D);
}

difference() {
    plate_body();
    holes();
}

// Printing notes:
//   - lay it flat on the bed, layers in the plane of the plate
//   - 4 perimeters, >= 40% infill; the load is in-plane tension/compression
//   - do NOT replace the hole rows by a slot: with oscillating controls the
//     joint reverses constantly and a slot develops play, which corrupts
//     theta1 / theta2 -- exactly the states that feed z4 and z5
