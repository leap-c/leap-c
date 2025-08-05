import atexit
import shutil

from pathlib import Path


class DeleteDirectoryHook:
    """A class that registers a cleanup hook to delete a directory.

    This class ensures a specified directory is removed automatically upon
    program termination, even if an error occurs. It uses `atexit.register`
    to queue the directory deletion.
    """

    def __init__(
        self,
        obj,
        dir: str | Path,
    ):
        """Initializes the DeleteDirectoryHook and registers the cleanup
        function.

        Args:
            obj: The object to which this hook is associated. A reference to
                this hook is stored on this object.
            dir: The path to the directory that should be deleted
                when the program exits.
        """
        self.dir = Path(dir)
        atexit.register(self.__del__)
        # Store a reference to this hook on the object
        obj.__delete_dir_hook = self

    def __del__(self):
        """Deletes the directory when the object is garbage collected or at
        exit.

        This method is registered with `atexit`. It attempts to remove the
        directory. Errors during deletion are ignored.
        """
        shutil.rmtree(self.dir, ignore_errors=True)
