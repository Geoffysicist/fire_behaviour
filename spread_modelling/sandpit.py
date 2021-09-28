from datetime import datetime as dt

now = dt.now()
print(now.hour)

    # detrmine moisture content
    # ros_df[MC] = np.where(
    #     ros_df[DATETIME].hour > 9 and ros_df[DATETIME].hour <= 20,
    #     np.where(
    #         ros_df[DATETIME].hour > 12 and ros_df[DATETIME].hour <= 17,
    #         2.76 + (0.124*ros_df[RH]) - (0.0187*ros_df[TEMP]),
    #         3.6 + (0.169*ros_df[RH]) - (0.045*ros_df[TEMP])
    #     ),
    #     3.08 + (0.198*ros_df[RH]) - (0.0483*ros_df[TEMP])
    # )
    
    # ros_df[FROS] = 0.0012*ros_df[FFDI]*fuel_load
    # ros_df[ROS] = 0.0012*get_FFDI(weather_df, flank=True)*fuel_load

    # return post_process(ros_df)