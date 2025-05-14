extends Node

var camera : CVCamera = CVCamera.new()
@export_group("Canvas")
@export var camera_canvas : Sprite2D;
var texture : ImageTexture;
var timer_framerate: Timer;
var image_type : int;
@export var video_path : String 

func _ready():
	#camera.open(0, 1920, 1080);
	var path = video_path
	camera.open_file(path);
	
	camera.flip(false, false);
	texture = ImageTexture.new();
	
	timer_framerate = Timer.new();
	timer_framerate.one_shot = false;
	timer_framerate.timeout.connect(on_timer_framerate);
	add_child(timer_framerate);
	var framerate = camera.get_framerate();
	print("framerate: ",  framerate,  ", ticker interval: ", 1 / framerate);
	timer_framerate.start(1 / framerate);
	
	
func on_timer_framerate():
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
