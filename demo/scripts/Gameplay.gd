extends Node

var camera : CVCamera = CVCamera.new()
@export_group("Canvas")
@export var camera_canvas : Sprite2D;
@export var overlay_canvas : Sprite2D
var texture : ImageTexture;
var overlay_texture : ImageTexture
var marker_textures : Array[ImageTexture]
var marker_canvases : Array[ImageTexture]
@export var marker_canvas_container : TabContainer
var timer_framerate: Timer;
var image_type : int;
@export_file("*.mp4") var video_path : String 

func _ready():
	#camera.open(0, 1920, 1080);
	var path = ""
	if OS.has_feature("editor"):
		path = ProjectSettings.globalize_path(video_path)
	else:
		path = OS.get_executable_path().get_base_dir().path_join(video_path)
	camera.open_file(path);
	
	camera.flip(false, false);
	texture = ImageTexture.new();

	overlay_texture = ImageTexture.new()
	
	# Setup playback
	timer_framerate = Timer.new();
	timer_framerate.one_shot = false;
	timer_framerate.timeout.connect(on_timer_framerate);
	add_child(timer_framerate);
	var framerate = camera.get_framerate();
	print("framerate: ",  framerate,  ", ticker interval: ", 1 / framerate);
	timer_framerate.start(1 / framerate);
	
	
func on_timer_framerate() -> void:
	match image_type:
		Enums.ImageType.RGB:
			texture.set_image(camera.get_image());
		Enums.ImageType.GRAYSCALE:
			texture.set_image(camera.get_greyscale_image())
		Enums.ImageType.THRESHOLD:
			texture.set_image(camera.get_threshold_image())
		_:
			printerr("Unreckognized image type")
			pass
	camera_canvas.texture = texture;
	camera.get_threshold_image()
	camera.find_rectangles()
	overlay_texture.set_image(camera.get_overlay_image())
	overlay_canvas.texture = overlay_texture
	draw_markers()
	
func draw_markers() -> void:
	var frames = camera.get_marker_frames()
	for child in marker_canvas_container.get_children():
		child.free()
	for marker in frames:
		var canvas = TextureRect.new()
		marker_canvas_container.add_child(canvas)
		var tex = ImageTexture.create_from_image(marker)
		canvas.custom_minimum_size = Vector2(200, 200)
		canvas.texture = tex
	pass

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("video_pause"):
		toggle_video_playback()
	if (Input.is_action_pressed("video_frame_forward")):
		frame_forward()
	if (Input.is_action_pressed("video_frame_backward")):
		frame_backward()
	pass

func _process(delta: float) -> void:
	pass
	
# Signals
func _update_threshold_type(type : int):
	camera.update_thres_type(type)
	pass

func _update_threshold_value(thres_val : float):
	camera.update_threshold_value(thres_val)
	pass

func _update_threshold_max_value(max_val : float):
	camera.update_threshold_max_value(max_val)
	pass

func _update_image_type(type : int):
	image_type = type
	pass

func _update_block_size(size : float):
	camera.update_threshold_blocksize(size)
	pass

func _update_threshold_c(c : float):
	camera.update_threshold_c(c)
	pass

func _update_adaptive_thresh_type(type : int):
	camera.update_threshold_adaptive_type(type)
	pass
	
func toggle_video_playback() -> void:
	timer_framerate.paused = (!timer_framerate.paused) 
	pass
	
func frame_forward() -> void:
	timer_framerate.paused = true;
	camera.frame_forward()
	on_timer_framerate()
	pass
	
func frame_backward() -> void:
	timer_framerate.paused = true;
	camera.frame_backward()
	on_timer_framerate()
	pass
