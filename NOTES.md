# draw a bounding box around the detected result and display the image
cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
# crop = final_image[y : start_y + tW, x : end_x + tH]
# cv2.imshow("Fianl", crop)
# cv2.imshow("Image", final)

# circle_img = cv2.circle(
#     original_image,
#     (x, y),
#     1,
#     (255, 255, 255),
#     thickness=-1,
# )

# cv2.imshow(
#     "x",
#     cv2.circle(
#         original_image,
#         (start_x, end_x),
#         1,
#         (255, 255, 255),
#         thickness=-1,
#     ),
# )

# cv2.imshow(
#     "y",
#     cv2.circle(
#         original_image,
#         (start_y, end_y),
#         1,
#         (255, 255, 255),
#         thickness=-1,
#     ),
# )

# cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

x = start_x + (round(tW * 0.5))
y = start_y + (round(tH * 0.5))
r = tW / 2

print(x, y, r)

# mask = np.zeros((tW, tH), dtype=np.uint8)
# cv2.circle(final, (x, y), round(r), (255, 255, 255), thickness=-1)
# masked_data = cv2.bitwise_and(final, final, mask=mask)
# crop = final[y : start_y + tW, x : end_x + tH]
# cv2.imshow("Fianl", crop)
# _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
# crop_img = cv2.img[y : y + 2 * r, x : (x + 2 * r)]

# _final = final[start_x:start_y, end_x:end_y]
# try:
#     cv2.imshow("cropped", _final)
# except Exception as e:
#     print(e)

# cv2.rectangle(final, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)
# cv2.imwrite("final.png", final)
# cv2.imshow("the final final", final)
cv2.waitKey(0)
