Registering new datasets
========================

Adding new datasets
-------------------

With hipct-data-tools installed, run the :file:`scripts/update_inventory.py`
script. You will probably want to update the ``keep_dataset()`` function
to limit which datasets get added to the registration inventory here.

Selecting common points
-----------------------

1. Generate a CSV file with neuroglancer links and copy to google sheets.
   This can be done using :file:`scripts/gen_common_point_spreadsheet.py`.
2. Use the generated CSV file to open two neuroglancer windows side by side,
   and find a common point in each dataset. This needs to be as accurate as
   possible for the registration to work later. Once you've done this, record
   the coordinates for each dataset in the CSV file.
3. With common points added to the CSV file, run
   :file:`scripts/read_common_point_spreadsheet.py`. This will read in the common
   points and add them to the internal registration database. At this point commit
   the changes to the internal database so they aren't lost.

Running registration
--------------------

Running :file:`scripts/register_inventory.py` will present you with a widget for
registering data. Before running this, edit the ``hipct_reg.data.STORAGE_DIR``
variable to point to where you want to save data locally.

When the widget opens, select a dataset you want to register. This will download
the data needed for registration, and then run the registration. The results
will be shown in three panels, which show:

1. The original ROI dataset.
2. The registered ROI dataset (in the registered coordinates).
3. The original full-organ dataset.

Compare panels 2 and 3. If the registration looks good, hit the "Good" button
and the registration parameters will be save to the registration inventory.
If the registration looks bad, type a reason in the notes text entry and
hit the "Bad" button. This will save the notes in the registration inventory.

You can keep going, registering multiple datasets with the widget open. Once done
make sure to commit the updates to the registration inventory.
