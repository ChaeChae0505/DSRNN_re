
```python
def robot_circle(robot_x, robot_y, r):
	#
  return

# One person danzer zone
def danzer_zone(human_x, human_y, human_vx, human_vy, theta):
	MV = 상수
	rStatic = 상수
	PI = math.pi

	v = sqrt(human_vx**2 + human_vy**2)
	r = MV*v + rStatic
	theta = (11*PI / 6) * math.exp(-1.4*v) + (PI / 6)

	return human_x , human_y , r, theta 


```
