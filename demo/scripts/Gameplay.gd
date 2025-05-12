extends Node

var camera : CVCamera = CVCamera.new();
@export_group("Canvas")
@export var camera_canvas : Sprite2D;
var texture : ImageTexture;
var timer_framerate: Timer;

func _ready():
	#camera.open(0, 1920, 1080);
	var path = '/home/nyr/source/AR/MarkerCube.mp4'
	print(path)
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

func _process(delta):
	pass
	
func on_timer_framerate():
	texture.set_image(camera.get_image());
	camera_canvas.texture = texture;
