def duration_task(dist, w):
    
  d_sort = sorted(dist)
  timer = 0
  
  i = 0
  while w > 0:
        w -= i
        if w==0:
            return timer
        timer +=1
        while i <= len(d_sort)-1 and timer >= d_sort[i]:
            i += 1
        
  return timer
  
distances = [10,20,30]
work = 100

print(duration_task(distances,work))