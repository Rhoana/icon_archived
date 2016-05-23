define(['zlib', 'util'],
function( Zlib, Util )
{
  var Image = function(purpose, imageId, project)
  {
    this.id = imageId;
    this.project = project;
    this.url = '/' + purpose + '/' + this.id + '.tif';
    this.labels_url = '/annotate.' + this.id + '.' + this.project.id + '.getannotations'
    this.labels_save_url = '/annotate.' + this.id + '.' + this.project.id + '.saveannotations'
    this.segmentation_url = '/annotate.' + this.id + '.' + this.project.id + '.getsegmentation'
    this.status_url = '/annotate.' + this.id + '.' + this.project.id + '.getstatus'

    this.segmentation = {};
    this.annotations = [];  // array of arrays
    this.annoationsLoadedCallbacks = [];

  };

  Image.prototype.resetAnnotation = function()
  {
    var num_annotations = this.annotations.length;
    this.annotations = [];
    for( var i=0; i<num_annotations; i++) {
    	this.annotations.push( [] );
    }
  }

  Image.prototype.addAnnotation = function(index, x, y)
  {
    this.annotations[ index ].push(x);
    this.annotations[ index ].push(y);
  }

  Image.prototype.initialize = function()
  {
    for (var i=0; i< this.project.labels.length; i++) {
        this.annotations.push( [] )
    }

    this.loadAnnotations();
  }

  Image.prototype.registerAnnotationsLoadedCallback = function(callback)
  {
    this.annoationsLoadedCallbacks.push(callback);
  }

  Image.prototype.loadAnnotations = function()
  {
    Util.load_data(this.labels_url, this.onAnnotationsLoaded.bind(this) );
  }

  Image.prototype.onAnnotationsLoaded = function(res)
  {
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    var data = JSON.parse( window.atob(decompressed) ); // decode the string
    if (data.length > 0) {
        this.annotations = data;
    }
    for(var i=0; i<this.annoationsLoadedCallbacks.length; i++) {
      this.annoationsLoadedCallbacks[i]();
    }
  }

  Image.prototype.save = function(label) {
    input = JSON.stringify(this.annotations);
    Util.send_data(this.labels_save_url, 'id=' + this.id + ';annotations='+input);
  }

  return Image;
});
