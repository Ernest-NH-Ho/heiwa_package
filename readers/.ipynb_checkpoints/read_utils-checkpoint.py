# from dataclasses import dataclass


def read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end):
    x = row[x_start:x_end].replace(" ", "")
    y = row[y_start:y_end].replace(" ", "")
    z = row[z_start:z_end].replace(" ", "")
    return x, y, z

def read_coord_coarse(row, coords, resi, x_start, x_end, y_start, y_end, z_start, z_end):
    coords[resi-1, 0, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 1, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 2, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    return coords

def extractNumbers(row, idx, field):
    start, end = idx[field]
    return str(row[start:end].replace(" ", ""))
