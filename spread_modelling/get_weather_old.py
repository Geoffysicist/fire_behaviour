# def get_weather(fn):

#     months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#     timeZone = "+10:00"
#     time = []
#     wind_speed = []
#     wind_dir = []
#     temp = []
#     rel_hum = []
#     dew_temp = []
#     drought_fac = []

#     with open(fn) as f:
#         weather = csv.reader(f)
#         header = True
#         series_array = [temp, rel_hum, wind_dir, wind_speed, drought_fac]
        
#         for row in weather:
#             try:
#                 d = row[0].split(' ')
#                 date = dt.date(int(d[2]), int(months.index(d[1])+1), int(d[0]))
#                 weather_data = row[2:]
#                 time.append(f'{date.isoformat()}T{row[1]}:00{timeZone}')
#                 for i in range(len(series_array)):
#                     series_array[i].append(weather_data[i])


#             except IndexError: # not a date of type "DD Mmm YYYY"            
#                 pass

#         weather_data = {
#             'time':time,
#             'temperature': temp,
#             'humidity': rel_hum,
#             'wind_dir': wind_dir,
#             'wind_speed': wind_speed
#         }

#         return pd.DataFrame(weather_data)
            
#         for t in range(len(time)):
#             print(time[t], end='\t')
#             for r in range(len(series_array)):
#                 try:
#                     print(f"{series_array[r][t]}", end="\t")
#                 except IndexError:
#                     print("", end="\t")
#             print('')
