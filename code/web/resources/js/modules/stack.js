define(['util'], function(Util)
{
	var Image = function(x, y, score) {
		this.x = x;
		this.y = y;
		this.score = score;
		this.vertices = [];
		this.shadow = [];
		this.base = [];
		this.type = [];
		this.foreground = "rgba(0,0,0,1.0)";
		this.background = "rgba(255,255,255,0.0)";
		this.shadow = "rgba(0,0,0,1.0)";
		this.purpose = '';
		this.annotationid = '';
		this.projectid = '';
	};

	var Stack = function(containerId, canvasId, horizontal) {
		this.horoizontal = horizontal;
		this.containerId = containerId;
		this.canvasId = canvasId;
		this.images = [];
		this.hover_index = -1;
		this.selected_index = -1;
	};


	Stack.prototype.initHorizontal = function(images, start, projectid, purpose) {

		var max_images_page = 10;
		var image = start;
		// var max_images = Math.min(images.length-start, max_images_page);
		// n = Math.min(max_images, images.length);

		console.log('id: '+ this.canvasId);
		//console.log('n: '+ n);
		console.log('start:' + start);
		console.log(images);

		var parent  = document.getElementById(this.containerId);
		
		var toremove = [];
		var children = parent.children;
		for (var i = 0; i < parent.children.length; i++) {
			var child = parent.children[i];
			if (child.id != this.canvasId) {
				toremove.push( child );
			}
		}

		for (var i = 0; i < toremove.length; i++) {
			parent.removeChild( toremove[i] );
		}


		this.images = [];

		if (images.length == 0) return;

		var c = document.getElementById(this.canvasId);
		c.addEventListener('click', this.mouseclick.bind(this), false);
		c.addEventListener('mousemove', this.mousemove.bind(this), false);

		 var x = 0;
		 var y = 0;
		 //var n = 6;
		 
		 var i = 0;
		 var left = 0;
		 
		 var zindex = 110;

		 //var nImages = n;//Math.min(20, n);
		 var top = 0;//0*nImages;
		 var imgIdx = 1;
		 var hgap = 25;
		 var vgap = 0;
		 var j=start;

		 for (i=0; i<max_images_page && j<images.length; i++,j++) {
		 	
			var image_id = images[ i ].image_id;
			var score    = (images[ i ].training_score*100.0).toFixed(2);
			var segFile = images[ i ].segmentation_file;
			var annFile = images[ i ].annotation_file;
			var color = Util.percentToRGB( score );

			console.log('adding: ' + image_id);


		    var divTag = document.createElement('div');
		    divTag.style.left = left + 'px';
		    divTag.style.top = top + 'px';
		    divTag.style.cssFloat = "left";
		    divTag.style.className = "stackelement";
		    divTag.style.position = "absolute";
		    divTag.style.zIndex = zindex;
		    divTag.setAttribute('id', image_id)

		    img = new Image(x, y+(i*10), 0.0);
		    img.purpose = purpose;
		    img.projectid = projectid;
		    img.annotationid = image_id;

		    var lower = 1;
		    var upper = 10;
		    imgIdx = parseInt((Math.random() * (upper - lower + 1)), 10) + lower;

		    var imgTag = document.createElement('img');
		    	

		    //if (i<4) {
		    if (segFile != null) {
		    	imgTag.src = 'images/seg' + imgIdx + '.png'; 
		    
		    }   
		    //else if (i<7) {
		    else if (annFile != null) {
		    	imgTag.src = 'images/ann' + imgIdx + '.png'; 
		 
		    } 
		    else {
		    	imgTag.src = 'images/grey' + imgIdx + '.png'; 
		    }
		    img.typecolor = "rgba(0,0,0,0.450)";
		    imgTag.className = 'stackimage';
		    //imgTag.style.border = "1px solid #0000FF";
		    divTag.appendChild(imgTag);
		    parent.appendChild(divTag);
		    zindex -=1;
		    left += hgap;
		    top += vgap;

		 	// console.log("i: " + i + " x: " + x + " y: " + y + ' left: '  + left);
		 	// console.log(divTag);

	 		img.type.push( {x: x+ (i*hgap) + 182, y: y + (vgap*i) + 0} );
	 		img.type.push( {x: x+ (i*hgap) + 185, y: y + (vgap*i) + 0} );
	 		img.type.push( {x: x+ (i*hgap) + 152, y: y + (vgap*i) + 218} );
	 		img.type.push( {x: x+ (i*hgap) + 149, y: y + (vgap*i) + 218} );

		 	if (i > 0) {
			 	img.vertices.push( {x: x+ (i*hgap) + 182-hgap, y: y + (vgap*i) + 5} );
		 		img.vertices.push( {x: x+ (i*hgap) + 182, y: y + (vgap*i) + 0} );
		 		img.vertices.push( {x: x+ (i*hgap) + 150, y: y + (vgap*i) + 219} );
		 		img.vertices.push( {x: x+ (i*hgap) + 38, y: y + (vgap*i) + 221} );
		 		img.vertices.push( {x: x+ (i*hgap) + 38, y: y + (vgap*i) + 221-vgap} );
		 		img.vertices.push( {x: x+ (i*hgap) + 150-hgap, y: y + (vgap*i) + 221-vgap} );
		 	}
		 	else {
			 	img.vertices.push( {x: x+ (i*hgap) + 72, y: y + 32} );
		 		img.vertices.push( {x: x+ (i*hgap) + 182, y: y + 0} );
		 		img.vertices.push( {x: x+ (i*hgap) + 150, y: y + 219} );
		 		img.vertices.push( {x: x+ (i*hgap) + 38, y: y + 221} );

		 	}

		 	this.images.push( img );

		 }

		 console.log(this.images);
	};	


	Stack.prototype.getMouse = function (e) {
	    var w = window, b = document.body;
	    return {x: e.clientX + (w.scrollX || b.scrollLeft || b.parentNode.scrollLeft || 0),
	    y: e.clientY + (w.scrollY || b.scrollTop || b.parentNode.scrollTop || 0)};
	};

	Stack.prototype.isPointInPoly = function(vertices, pt){

			var parent  = document.getElementById(this.containerId);
			var rect = parent.getBoundingClientRect();
			//sconsole.log(rect);
			// console.log(rect);
			// this.offset = {x: parent.style.left, y: parent.style.top};

		//var offset  = {x: rect.left, y: rect.top};
		var offset  = {x:parent.offsetLeft , y:parent.offsetTop};

	    for(var c = false, i = -1, l = vertices.length, j = l - 1; ++i < l; j = i) {

	    	var vi = {x: vertices[i].x + offset.x, y: vertices[i].y + offset.y};
	    	var vj = {x: vertices[j].x + offset.x, y: vertices[j].y + offset.y};

	        ((vi.y <= pt.y && pt.y < vj.y) || (vj.y <= pt.y && pt.y < vi.y))
	        && (pt.x < (vj.x - vi.x) * (pt.y - vi.y) / (vj.y - vi.y) + vi.x)
	        && (c = !c);
	    }

	    return c;
	};		

	Stack.prototype.mousemove = function(e) {

		//console.log('Stack.prototype.mousemove..');
		var stackElt = document.getElementById(this.containerId);

		var mouse = this.getMouse(e);

		var parent  = document.getElementById(this.containerId);
		//var rect = parent.getBoundingClientRect();
		// var offset  = {x: rect.left, y: rect.top};
		var offset  = {x:stackElt.offsetLeft , y:stackElt.offsetTop};

		//var delta = {x: mouse.x - rect.left , y: mouse.y-rect.top};
		var delta = {x: mouse.x - stackElt.offsetLeft , y: mouse.y-stackElt.offsetTop};
		var eoffset = {x:stackElt.offsetLeft , y:stackElt.offsetTop};
		

		// //console.log('onmousemove..:' + this.mouse_on_stack);
		// document.getElementById('mouse').innerHTML = "mouse (" + mouse.x + "," +  mouse.y + ")";
		// if (this.images.length > 0) {
		// 	document.getElementById('image').innerHTML = "image (" + this.images[0].vertices[0].x + "," +  this.images[0].vertices[0].y + ")";			
		// }
		// document.getElementById('delta').innerHTML = "delta (" + delta.x + "," +  delta.y + ")";
		// document.getElementById('eoffset').innerHTML = "eoffset (" + eoffset.x + "," +  eoffset.y + ")";


		 var new_hover_index = -1;
		 var i, j;
		 for (i=0; i<this.images.length; i++) {
		 	img = this.images[i];

		 	if (this.isPointInPoly(img.vertices, mouse))
		 	{
		 		new_hover_index = i;
		 		//console.log('image: '+i);
		 		break;
		 	}
		}

		if (new_hover_index != this.hover_index) {
			this.hover_index = new_hover_index;
			this.draw();
		}

	};

	Stack.prototype.mouseclick = function(e) {

		console.log('Stack.prototype.mouseclick..');

		if (this.hover_index != -1) {

			var mouse = this.getMouse(e);
		 	var img = this.images[this.hover_index];

		 	if (this.isPointInPoly(img.vertices, mouse))
		 	{
		 		console.log('onclick..yaass');
		 		this.selected_index = this.hover_index;

		 		//var element = e.srcElement;
		 		console.log(img);

		 		//this.draw();
		 		window.location = '/annotate' + '.' + img.purpose + '.' + img.annotationid + '.' + img.projectid;

		 	}			
		}
		
	};

	Stack.prototype.drawPolygon = function(ctx, vertices, foreground, background, shadow) {

		if (vertices.length == 0) return;
		//if (shadow) 
		{
			 ctx.shadowBlur=6;
			 ctx.shadowOffsetX = 3;
			 ctx.shadowOffsetY = 3;
			 ctx.shadowColor=shadow;				
		}

		ctx.beginPath();
		ctx.lineWidth = 1;
		ctx.lineCap = 'round';
		ctx.strokeStyle = foreground;
		ctx.fillStyle = background;
		for(j=0; j<vertices.length; j++) {
			ctx.lineTo( vertices[j].x, vertices[j].y );
		}
		ctx.lineTo( vertices[0].x, vertices[0].y );
		ctx.closePath();
		ctx.fillStyle = background;
		ctx.fill();
		ctx.stroke();



	};

	Stack.prototype.draw = function(e) {

	 	var c = document.getElementById(this.canvasId);

	 	//console.log('this.canvasId: ' + this.canvasId);
		var ctx = c.getContext("2d");
		ctx.clearRect(0, 0, c.width, c.height);
		
		var i, j;

		for (i=this.images.length-1; i>=0; i--) {
			img = this.images[i];
			if (i==this.hover_index || i==this.selected_index) {
				this.drawPolygon(ctx, img.vertices, img.foreground, "rgba(255,255,0,0.45)",  img.shadow );
			}
			else {
				this.drawPolygon(ctx, img.vertices, img.foreground, img.background,  img.shadow );
			}

			this.drawPolygon(ctx, img.type, img.foreground, img.typecolor, img.shadow );

		}

	};

	return Stack;
});
	
